//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/program_manager.hpp>

#include <sycl/__impl/exception.hpp>

#include <detail/device_impl.hpp>
#include <detail/offload/offload_utils.hpp>

#include <cstring>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

static inline bool checkFatBinVersion(const __sycl_tgt_bin_desc &FatbinDesc) {
  return FatbinDesc.Version == SupportedOffloadBinaryVersion;
}

static inline bool
checkDeviceImageValidity(const __sycl_tgt_device_image &DeviceImage) {
  return (DeviceImage.Version == SupportedDeviceBinaryVersion) &&
         (DeviceImage.OffloadKind == llvm::object::OFK_SYCL) &&
         (DeviceImage.ImageFormat == llvm::object::IMG_SPIRV);
}

void ProgramAndKernelManager::registerFatBin(__sycl_tgt_bin_desc *FatbinDesc) {
  assert(FatbinDesc && "Device images descriptor can't be nullptr");

  if (!checkFatBinVersion(*FatbinDesc))
    throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                          "Incompatible version of device images descriptor.");
  if (!FatbinDesc->NumDeviceBinaries)
    return;

  std::lock_guard<std::mutex> Guard(MDataCollectionMutex);
  for (uint16_t I = 0; I < FatbinDesc->NumDeviceBinaries; ++I) {
    const auto &RawDeviceImage = FatbinDesc->DeviceImages[I];
    if (!checkDeviceImageValidity(RawDeviceImage))
      throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                            "Incompatible device image.");

    const llvm::offloading::EntryTy *EntriesB = RawDeviceImage.EntriesBegin;
    const llvm::offloading::EntryTy *EntriesE = RawDeviceImage.EntriesEnd;
    // Ignore "empty" device image.
    if (EntriesB == EntriesE)
      continue;

    std::unique_ptr<DeviceImageManager> NewImageWrapper =
        std::make_unique<DeviceImageManager>(RawDeviceImage);

    for (auto EntriesIt = EntriesB; EntriesIt != EntriesE; ++EntriesIt) {
      auto Name = EntriesIt->SymbolName;

      auto It = MDeviceKernelInfoMap.find(std::string_view(Name));
      if (It == MDeviceKernelInfoMap.end()) {

        [[maybe_unused]] auto [Iterator, EmplaceSucceeded] =
            MDeviceKernelInfoMap.emplace(
                std::piecewise_construct, std::forward_as_tuple(Name),
                std::forward_as_tuple(Name, *NewImageWrapper));
        assert(EmplaceSucceeded && "Kernel name found in multiple images");
      }
    }

    MDeviceImageManagers.insert(
        std::make_pair(&RawDeviceImage, std::move(NewImageWrapper)));
  }
}

void ProgramAndKernelManager::unregisterFatBin(
    __sycl_tgt_bin_desc *FatbinDesc) {
  assert(FatbinDesc && "Device images descriptor can't be nullptr");

  if (!checkFatBinVersion(*FatbinDesc) || FatbinDesc->NumDeviceBinaries == 0)
    return;

  std::lock_guard<std::mutex> Guard(MDataCollectionMutex);
  for (uint16_t I = 0; I < FatbinDesc->NumDeviceBinaries; ++I) {
    const auto &RawDeviceImage = FatbinDesc->DeviceImages[I];

    auto DevImageIt = MDeviceImageManagers.find(&RawDeviceImage);
    if (DevImageIt == MDeviceImageManagers.end())
      continue;

    const llvm::offloading::EntryTy *EntriesB = RawDeviceImage.EntriesBegin;
    const llvm::offloading::EntryTy *EntriesE = RawDeviceImage.EntriesEnd;
    // Ignore "empty" device image
    if (EntriesB == EntriesE)
      continue;

    for (auto EntriesIt = EntriesB; EntriesIt != EntriesE; ++EntriesIt) {
      if (auto KernelIt = MDeviceKernelInfoMap.find(EntriesIt->SymbolName);
          KernelIt != MDeviceKernelInfoMap.end()) {
        // Programs are attached to image and will be released with image
        // destruction. Clear only kernel specific data by destroying its kernel
        // info object.
        MDeviceKernelInfoMap.erase(KernelIt);
      }
    }

    MDeviceImageManagers.erase(DevImageIt);
  }
}

static bool isImageCompatible(const DeviceImageManager &Image,
                              const DeviceImpl &Device) {
  sycl::backend BE = Device.getBackend();
  const char *Target = Image.getRawData().TripleString;

  if (!(strcmp(Target, DeviceBinaryTripleSPIRV64) == 0 &&
        BE == sycl::backend::level_zero))
    return false;

  bool IsValid{};
  callAndThrow(olIsValidBinary, Device.getOLHandle(),
               Image.getRawData().ImageStart, Image.getSize(), &IsValid);
  return IsValid;
}

ol_symbol_handle_t
ProgramAndKernelManager::getOrCreateKernel(DeviceKernelInfo &KernelInfo,
                                           DeviceImpl &Device) {

  std::lock_guard<std::mutex> KernelGuard(MDataCollectionMutex);

  if (auto Kernel = KernelInfo.getKernel(Device.getOLHandle()))
    return Kernel;

  auto &DeviceImage = KernelInfo.getDeviceImage();

  if (!isImageCompatible(DeviceImage, Device))
    throw exception(make_error_code(errc::runtime),
                    std::string("No compatible image for ") +
                        KernelInfo.getName().data() + " was found");

  auto DeviceHandle = Device.getOLHandle();
  auto Program = DeviceImage.getOrCreateProgram(DeviceHandle);

  ol_symbol_handle_t Kernel{};
  callAndThrow(olGetSymbol, Program, KernelInfo.getName().data(),
               OL_SYMBOL_KIND_KERNEL, &Kernel);
  KernelInfo.addKernel(DeviceHandle, Kernel);
  return Kernel;
}

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL

extern "C" _LIBSYCL_EXPORT void
__sycl_register_lib(sycl::detail::__sycl_tgt_bin_desc *FatbinDesc) {
  sycl::detail::ProgramAndKernelManager::getInstance().registerFatBin(
      FatbinDesc);
}

extern "C" _LIBSYCL_EXPORT void
__sycl_unregister_lib(sycl::detail::__sycl_tgt_bin_desc *FatbinDesc) {
  sycl::detail::ProgramAndKernelManager::getInstance().unregisterFatBin(
      FatbinDesc);
}
