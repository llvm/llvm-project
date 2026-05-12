//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the class for kernel and program
/// management.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_PROGRAM_MANAGER
#define _LIBSYCL_PROGRAM_MANAGER

#include <sycl/__impl/detail/config.hpp>

#include <detail/device_binary_structures.hpp>
#include <detail/device_image_wrapper.hpp>
#include <detail/device_kernel_info.hpp>

#include <llvm/Object/OffloadBinary.h>

#include <OffloadAPI.h>

#include <mutex>
#include <unordered_map>
#include <vector>

// +++ Entry points referenced by the offload wrapper object {

/// Executed as a part of a module's (.exe, .dll) static initialization.
/// Registers device executable images with the runtime.
/// \param BinaryStart pointer to the start of the OffloadBinary.
/// \param Size size in bytes of the OffloadBinary.
extern "C" _LIBSYCL_EXPORT void __sycl_register_lib(const void *BinaryStart,
                                                    size_t Size);

/// Executed as a part of current module's (.exe, .dll) static
/// de-initialization.
/// Unregisters device executable images with the runtime.
/// \param BinaryStart pointer to the start of the OffloadBinary.
/// \param Size size in bytes of the OffloadBinary.
/// BinaryStart and Size must match the values passed to the corresponding
/// __sycl_register_lib call.
extern "C" _LIBSYCL_EXPORT void __sycl_unregister_lib(const void *BinaryStart,
                                                      size_t Size);

// +++ }

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

class DeviceImpl;

/// A class to manage programs and kernels.
class ProgramAndKernelManager {

public:
  static ProgramAndKernelManager &getInstance() {
    static ProgramAndKernelManager PM{};
    return PM;
  }

  /// Parses the OffloadBinary of the given Size starting at BinaryStart and
  /// prepares internal structures for effective kernel/program creation.
  /// \throw sycl::exception with sycl::errc::runtime if parsing fails or if
  /// the binary has an incompatible kind or target.
  void registerFatBin(const void *BinaryStart, size_t Size);

  /// Removes all entries associated with the fat binary that was previously
  /// passed to registerFatBin. BinaryStart and Size must match the values
  /// passed to the corresponding registerFatBin call.
  void unregisterFatBin(const void *BinaryStart, size_t Size);

  /// Creates a liboffload kernel that is ready for execution.
  /// This method is thread-safe.
  /// \param KernelInfo a set of kernel specific data: name, corresponding
  /// device image, etc.
  /// \param Device the device for which this kernel must be compiled.
  /// \return a liboffload kernel handle that is ready to be passed to kernel
  /// execution methods.
  ol_symbol_handle_t getOrCreateKernel(DeviceKernelInfo &KernelInfo,
                                       DeviceImpl &Device);

private:
  ProgramAndKernelManager() = default;
  ~ProgramAndKernelManager() = default;
  ProgramAndKernelManager(ProgramAndKernelManager const &) = delete;
  ProgramAndKernelManager &operator=(ProgramAndKernelManager const &) = delete;

  // Filled by registerFatBin(...).
  // Map for storing device kernel information. Runtime lookup should be avoided
  // by caching the pointers when possible.
  std::unordered_map<std::string_view, DeviceKernelInfo> MDeviceKernelInfoMap;

  // Keyed by BinaryStart (register/unregister param). Each fat binary can
  // contain multiple device images, each owned by its own DeviceImageManager.
  // Controls lifetime of device image managers and, through them, parsed
  // OffloadBinary objects.
  using BinaryStartKey = const void *;
  using DeviceImageManagerVec =
      std::vector<std::unique_ptr<DeviceImageManager>>;
  std::unordered_map<BinaryStartKey, DeviceImageManagerVec>
      MDeviceImageManagers;

  // All work with device images and data related to it must be wrapped with a
  // lock of this mutex.
  std::mutex MDataCollectionMutex;
};

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_PROGRAM_MANAGER
