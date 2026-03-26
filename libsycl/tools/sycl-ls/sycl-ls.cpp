//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The "sycl-ls" utility lists all platforms and devices discovered by SYCL.
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

std::string getDeviceTypeName(const device &Device) {
  auto DeviceType = Device.get_info<info::device::device_type>();
  switch (DeviceType) {
  case info::device_type::cpu:
    return "cpu";
  case info::device_type::gpu:
    return "gpu";
  case info::device_type::host:
    return "host";
  case info::device_type::accelerator:
    return "accelerator";
  default:
    return "unknown";
  }
}

static void printDeviceInfo(const device &Device, bool Verbose,
                            const std::string &Prepend) {
  auto DeviceName = Device.get_info<info::device::name>();
  auto DeviceVendor = Device.get_info<info::device::vendor>();
  auto DeviceDriverVersion = Device.get_info<info::device::driver_version>();

  if (Verbose) {
    std::cout << Prepend << "Type              : " << getDeviceTypeName(Device)
              << std::endl;
    std::cout << Prepend << "Name              : " << DeviceName << std::endl;
    std::cout << Prepend << "Vendor            : " << DeviceVendor << std::endl;
    std::cout << Prepend << "Driver            : " << DeviceDriverVersion
              << std::endl;
  } else {
    std::cout << Prepend << ", " << DeviceName << " [" << DeviceDriverVersion
              << "]" << std::endl;
  }
}

static void
printSelectorChoice(const detail::DeviceSelectorInvocableType &Selector,
                    const std::string &Prepend) {
  try {
    const auto &Device = device(Selector);
    std::string DeviceTypeName = getDeviceTypeName(Device);
    auto Platform = Device.get_info<info::device::platform>();
    auto PlatformName = Platform.get_info<info::platform::name>();
    printDeviceInfo(Device, false /*Verbose*/,
                    Prepend + DeviceTypeName + ", " + PlatformName);
  } catch (const sycl::exception &Exception) {
    std::string What = Exception.what();
    constexpr size_t MaxLength = 80;
    // Truncate long string so it can fit in one-line
    if (What.length() > MaxLength)
      What = What.substr(0, MaxLength) + "...";
    std::cout << Prepend << What << std::endl;
  }
}

int main(int argc, char **argv) {
  llvm::cl::opt<bool> Verbose(
      "verbose", llvm::cl::desc("Verbosely prints all the discovered devices"));
  llvm::cl::alias VerboseShort("v", llvm::cl::desc("Alias for -verbose"),
                               llvm::cl::aliasopt(Verbose));
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "This program lists all backends and devices discovered by SYCL");

  try {
    const auto &Platforms = platform::get_platforms();

    if (Platforms.size() == 0) {
      std::cout << "No platforms found." << std::endl;
      return EXIT_SUCCESS;
    }

    for (const auto &Platform : Platforms) {
      backend Backend = Platform.get_backend();
      auto PlatformName = Platform.get_info<info::platform::name>();
      const auto &Devices = Platform.get_devices();

      for (const auto &Device : Devices) {
        std::cout << "[" << getBackendName(Backend) << ":"
                  << getDeviceTypeName(Device) << "]";
        std::cout << " ";
        // Verbose parameter is set to false to print regular devices output
        // first
        printDeviceInfo(Device, false, PlatformName);
      }
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

        const auto &Devices = Platform.get_devices();
        std::cout << "    Devices  : " << Devices.size() << std::endl;
        for (const auto &Device : Devices) {
          printDeviceInfo(Device, true, "        ");
        }
      }

      // Print built-in device selectors choice
      printSelectorChoice(default_selector_v, "default_selector()      : ");
      printSelectorChoice(accelerator_selector_v, "accelerator_selector()  : ");
      printSelectorChoice(cpu_selector_v, "cpu_selector()          : ");
      printSelectorChoice(gpu_selector_v, "gpu_selector()          : ");
    }
  } catch (sycl::exception &e) {
    std::cerr << "SYCL Exception encountered: " << e.what() << std::endl
              << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
