//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/program_manager.hpp>

#include <sycl/__impl/detail/get_device_kernel_info.hpp>
#include <sycl/__impl/exception.hpp>

#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/offload/offload_utils.hpp>

#include <llvm/Frontend/Offloading/Utility.h>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

_LIBSYCL_EXPORT DeviceKernelInfo &
getDeviceKernelInfo(std::string_view KernelName) {
  return ProgramAndKernelManager::getInstance().getDeviceKernelInfo(KernelName);
}

DeviceKernelInfo &
ProgramAndKernelManager::getDeviceKernelInfo(std::string_view KernelName) {
  auto It = MDeviceKernelInfoMap.find(KernelName);
  assert(It != MDeviceKernelInfoMap.end());
  return It->second;
}

void ProgramAndKernelManager::releaseResources() {
  std::lock_guard<std::mutex> Guard(MDataCollectionMutex);
  // Contexts can outlive this call: platform default contexts are kept in the
  // platform cache, which is static. Programs must not be left for
  // their destructors to release, because olShutDown() follows this call.
  for (const std::weak_ptr<ContextImpl> &WeakContext : MContextsWithPrograms) {
    if (std::shared_ptr<ContextImpl> Context = WeakContext.lock())
      Context->releaseAllPrograms();
  }
  MContextsWithPrograms.clear();
  MDeviceKernelInfoMap.clear();
  MDeviceImageManagers.clear();
}

void ProgramAndKernelManager::trackContext(
    const std::shared_ptr<ContextImpl> &Context) {
  bool AlreadyTracked = false;
  for (auto It = MContextsWithPrograms.begin();
       It != MContextsWithPrograms.end();) {
    std::shared_ptr<ContextImpl> TrackedContext = It->lock();
    if (!TrackedContext) {
      // Remove expired context from the tracking list.
      It = MContextsWithPrograms.erase(It);
      continue;
    }
    AlreadyTracked |= (TrackedContext == Context);
    ++It;
  }

  if (!AlreadyTracked)
    MContextsWithPrograms.push_back(Context);
}

static inline bool
checkDeviceImageValidity(const llvm::object::OffloadBinary &OB) {
  return (OB.getOffloadKind() == llvm::object::OFK_SYCL) &&
         (OB.getImageKind() == llvm::object::IMG_SPIRV);
}

void ProgramAndKernelManager::registerFatBin(const void *BinaryStart,
                                             size_t Size) {
  assert(BinaryStart && "Binary pointer can't be nullptr");

  llvm::MemoryBufferRef MBR(
      llvm::StringRef(static_cast<const char *>(BinaryStart), Size),
      /*Identifier=*/"");
  auto BinOrErr = llvm::object::OffloadBinary::create(MBR);
  if (!BinOrErr) {
    throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                          "Failed to parse OffloadBinary: " +
                              llvm::toString(BinOrErr.takeError()));
  }
  assert(!BinOrErr->empty() && "OffloadBinary must contain at least one entry");

  DeviceImageManagerVec Images;
  Images.reserve(BinOrErr->size());

  std::lock_guard<std::mutex> Guard(MDataCollectionMutex);
  for (std::unique_ptr<llvm::object::OffloadBinary> &OB : *BinOrErr) {
    if (!checkDeviceImageValidity(*OB))
      throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                            "Incompatible device image.");

    llvm::StringRef Symbols = OB->getString("symbols");

    Images.push_back(std::make_unique<DeviceImageManager>(std::move(OB)));
    DeviceImageManager &NewImageWrapper = *Images.back();

    llvm::offloading::sycl::forEachSymbol(Symbols, [&](llvm::StringRef Name) {
      auto It = MDeviceKernelInfoMap.find(std::string_view(Name));
      if (It == MDeviceKernelInfoMap.end()) {
        [[maybe_unused]] auto [Iterator, EmplaceSucceeded] =
            MDeviceKernelInfoMap.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(std::string_view(Name)),
                std::forward_as_tuple(std::string_view(Name), NewImageWrapper));
        assert(EmplaceSucceeded && "Kernel name found in multiple images");
      }
    });
  }

  [[maybe_unused]] auto [It, Inserted] =
      MDeviceImageManagers.emplace(BinaryStart, std::move(Images));
  assert(Inserted && "Fat binary registered twice");
}

void ProgramAndKernelManager::unregisterFatBin(const void *BinaryStart,
                                               size_t /*Size*/) {
  assert(BinaryStart && "Binary pointer can't be nullptr");

  std::lock_guard<std::mutex> Guard(MDataCollectionMutex);
  auto It = MDeviceImageManagers.find(BinaryStart);
  if (It == MDeviceImageManagers.end())
    return;

  for (auto &Image : It->second) {
    // Programs created from this image are owned by the contexts they were
    // created in, so they have to be destroyed here: the image is about to go
    // away, and the kernel names cached alongside those programs point into its
    // memory, which may be unmapped right after this call.
    for (const std::weak_ptr<ContextImpl> &WeakContext :
         MContextsWithPrograms) {
      if (std::shared_ptr<ContextImpl> Context = WeakContext.lock())
        Context->releaseProgramsForImage(*Image);
    }

    llvm::StringRef Symbols = Image->getOffloadBinary().getString("symbols");
    llvm::offloading::sycl::forEachSymbol(Symbols, [&](llvm::StringRef Name) {
      if (auto KernelIt = MDeviceKernelInfoMap.find(std::string_view(Name));
          KernelIt != MDeviceKernelInfoMap.end()) {
        // Clear kernel specific data by destroying its kernel info object.
        MDeviceKernelInfoMap.erase(KernelIt);
      }
    });
  }
  MDeviceImageManagers.erase(It);
}

static bool isImageCompatible(const DeviceImageManager &Image,
                              const DeviceImpl &Device) {
  const llvm::object::OffloadBinary &OB = Image.getOffloadBinary();
  if (!(OB.getTriple() == DeviceBinaryTripleSPIRV64 &&
        Device.getBackend() == sycl::backend::level_zero))
    return false;

  bool IsValid{};
  llvm::StringRef ImageBytes = OB.getImage();
  callAndThrow(olIsValidBinary, Device.getOLHandle(), ImageBytes.data(),
               ImageBytes.size(), &IsValid);
  return IsValid;
}

ol_symbol_handle_t ProgramAndKernelManager::getOrCreateKernel(
    DeviceKernelInfo &KernelInfo, const std::shared_ptr<ContextImpl> &Context,
    DeviceImpl &Device) {
  assert(Context && "Context can't be nullptr");

  std::lock_guard<std::mutex> KernelGuard(MDataCollectionMutex);

  DeviceImageManager &DeviceImage = KernelInfo.getDeviceImage();

  if (!isImageCompatible(DeviceImage, Device))
    throw exception(make_error_code(errc::runtime),
                    std::string("No compatible image for ") +
                        KernelInfo.getName().data() + " was found");

  // Track the context before it caches anything, so that unregisterFatBin() can
  // reach the programs it is about to create.
  trackContext(Context);

  // Lock order is MDataCollectionMutex -> ContextImpl::MProgramCacheMutex.
  return Context->getOrCreateKernel(DeviceImage, Device.getOLHandle(),
                                    KernelInfo.getName());
}

bool ProgramAndKernelManager::hasCompatibleImage(const DeviceImpl &Device) {
  std::lock_guard<std::mutex> Guard(MDataCollectionMutex);

  for (const auto &BinaryImagesPair : MDeviceImageManagers) {
    for (const auto &Image : BinaryImagesPair.second) {
      if (isImageCompatible(*Image, Device))
        return true;
    }
  }

  return false;
}

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL

extern "C" _LIBSYCL_EXPORT void __sycl_register_lib(const void *BinaryStart,
                                                    size_t Size) {
  sycl::detail::ProgramAndKernelManager::getInstance().registerFatBin(
      BinaryStart, Size);
}

extern "C" _LIBSYCL_EXPORT void __sycl_unregister_lib(const void *BinaryStart,
                                                      size_t Size) {
  sycl::detail::ProgramAndKernelManager::getInstance().unregisterFatBin(
      BinaryStart, Size);
}
