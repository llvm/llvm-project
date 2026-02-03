//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The "sycl-ls" utility lists all platforms discovered by SYCL.
//
// There are two types of output:
//   concise (default) and
//   verbose (enabled with --verbose).
//
#include <sycl/sycl.hpp>

#include "llvm/Support/CommandLine.h"

#include <iostream>

using namespace sycl;
using namespace std::literals;

inline std::string_view getBackendName(const backend &Backend) {
  switch (Backend) {
  case backend::opencl:
    return "opencl";
  case backend::level_zero:
    return "level_zero";
  case backend::cuda:
    return "cuda";
  case backend::hip:
    return "hip";
  }

  return "";
}

int main(int argc, char **argv) {
  llvm::cl::opt<bool> Verbose(
      "verbose",
      llvm::cl::desc("Verbosely prints all the discovered platforms"));
  llvm::cl::alias VerboseShort("v", llvm::cl::desc("Alias for -verbose"),
                               llvm::cl::aliasopt(Verbose));
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "This program lists all backends discovered by SYCL");

  try {
    const auto &Platforms = platform::get_platforms();

    if (Platforms.size() == 0) {
      std::cout << "No platforms found." << std::endl;
      return EXIT_SUCCESS;
    }

    for (const auto &Platform : Platforms) {
      backend Backend = Platform.get_backend();
      std::cout << "[" << getBackendName(Backend) << ":"
                << "unknown" << "]" << std::endl;
    }

    if (Verbose) {
      std::cout << "\nPlatforms: " << Platforms.size() << std::endl;
      uint32_t PlatformNum = 0;
      for (const auto &Platform : Platforms) {
        ++PlatformNum;
        auto PlatformVersion = Platform.get_info<info::platform::version>();
        auto PlatformName = Platform.get_info<info::platform::name>();
        auto PlatformVendor = Platform.get_info<info::platform::vendor>();
        std::cout << "Platform [#" << PlatformNum << "]:" << std::endl;
        std::cout << "    Version  : " << PlatformVersion << std::endl;
        std::cout << "    Name     : " << PlatformName << std::endl;
        std::cout << "    Vendor   : " << PlatformVendor << std::endl;
        std::cout << "    Devices  : " << "unknown" << std::endl;
      }
    }
  } catch (sycl::exception &e) {
    std::cerr << "SYCL Exception encountered: " << e.what() << std::endl
              << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
