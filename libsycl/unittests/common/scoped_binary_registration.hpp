//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// RAII helpers for registering and unregistering test device binaries.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_UNITTESTS_COMMON_SCOPED_BINARY_REGISTRATION_HPP
#define _LIBSYCL_UNITTESTS_COMMON_SCOPED_BINARY_REGISTRATION_HPP

#include <common/device_images.hpp>

#include <detail/program_manager.hpp>

#include <array>
#include <string>
#include <utility>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace unittests {

class ScopedBinaryRegistration {
public:
  explicit ScopedBinaryRegistration(llvm::ArrayRef<llvm::StringRef> KernelNames)
      : MBinary(createSYCLDeviceBinary(KernelNames)) {
    sycl::detail::ProgramAndKernelManager::getInstance().registerFatBin(
        MBinary.data(), MBinary.size());
  }

  ~ScopedBinaryRegistration() {
    sycl::detail::ProgramAndKernelManager::getInstance().unregisterFatBin(
        MBinary.data(), MBinary.size());
  }

private:
  llvm::SmallString<0> MBinary;
};

class ScopedKernelRegistration {
public:
  explicit ScopedKernelRegistration(std::string KernelName)
      : MKernelName(std::move(KernelName)),
        MRegistration(createKernelNames(MKernelName)) {}

private:
  static std::array<llvm::StringRef, 1>
  createKernelNames(const std::string &KernelName) {
    return {KernelName};
  }

  std::string MKernelName;
  ScopedBinaryRegistration MRegistration;
};

} // namespace unittests
_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_UNITTESTS_COMMON_SCOPED_BINARY_REGISTRATION_HPP
