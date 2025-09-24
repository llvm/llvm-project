//===- LevelZeroArch.cpp - list installed Level Zero devices ---*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a tool for detecting Level Zero devices installed in the
// system
//
//===----------------------------------------------------------------------===//

#ifndef HAVE_LEVEL_ZERO_HEADERS

int printGPUsByLevelZero() { return 0; }

#else

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include <cstdio>
#include <level_zero/ze_api.h>

using namespace llvm;
extern cl::opt<bool> Verbose;

#define DEFINE_WRAPPER(NAME)                                                   \
  using NAME##_ty = decltype(NAME);                                            \
  void *NAME##Ptr = nullptr;                                                   \
  template <class... Ts> ze_result_t NAME##_wrap(Ts... args) {                 \
    if (!NAME##Ptr) {                                                          \
      return ZE_RESULT_ERROR_UNKNOWN;                                          \
    }                                                                          \
    return reinterpret_cast<NAME##_ty *>(NAME##Ptr)(args...);                  \
  };

DEFINE_WRAPPER(zeInitDrivers)
DEFINE_WRAPPER(zeDeviceGet)
DEFINE_WRAPPER(zeDeviceGetProperties)

static bool loadLevelZero() {
  const char *L0Library = "libze_loader.so";
  std::string ErrMsg;

  auto DynlibHandle = std::make_unique<llvm::sys::DynamicLibrary>(
      llvm::sys::DynamicLibrary::getPermanentLibrary(L0Library, &ErrMsg));
  if (!DynlibHandle->isValid()) {
    if (ErrMsg.empty())
      ErrMsg = "unknown error";
    if (Verbose)
      llvm::errs() << "Unable to load library '" << L0Library << "': " << ErrMsg
                   << "\n";
    return false;
  }

  constexpr struct {
    const char *name;
    void **fptr;
  } dlwrap[] = {
      {"zeInitDrivers", &zeInitDriversPtr},
      {"zeDeviceGet", &zeDeviceGetPtr},
      {"zeDeviceGetProperties", &zeDeviceGetPropertiesPtr},
  };

  for (auto entry : dlwrap) {
    void *P = DynlibHandle->getAddressOfSymbol(entry.name);
    if (P == nullptr) {
      if (Verbose)
        llvm::errs() << "Unable to find '" << entry.name << "' in '"
                     << L0Library << "'\n";
      return false;
    }
    *(entry.fptr) = P;
  }

  return true;
}

#define CALL_ZE_AND_CHECK(Fn, ...)                                             \
  do {                                                                         \
    ze_result_t Rc = Fn##_wrap(__VA_ARGS__);                                   \
    if (Rc != ZE_RESULT_SUCCESS) {                                             \
      if (Verbose)                                                             \
        llvm::errs() << "Error: " << __func__ << ":" << #Fn                    \
                     << " failed with error code " << Rc << "\n";              \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int printGPUsByLevelZero() {
  if (!loadLevelZero())
    return 1;

  ze_init_driver_type_desc_t driver_type = {};
  driver_type.stype = ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC;
  driver_type.flags = ZE_INIT_DRIVER_TYPE_FLAG_GPU;
  driver_type.pNext = nullptr;
  uint32_t driverCount{0};

  // Initialize and find all drivers
  CALL_ZE_AND_CHECK(zeInitDrivers, &driverCount, nullptr, &driver_type);

  llvm::SmallVector<ze_driver_handle_t> drivers(driverCount);
  CALL_ZE_AND_CHECK(zeInitDrivers, &driverCount, drivers.data(), &driver_type);

  for (auto driver : drivers) {
    // Discover all the devices for a given driver
    uint32_t deviceCount = 0;
    CALL_ZE_AND_CHECK(zeDeviceGet, driver, &deviceCount, nullptr);

    llvm::SmallVector<ze_device_handle_t> devices(deviceCount);
    CALL_ZE_AND_CHECK(zeDeviceGet, driver, &deviceCount, devices.data());

    for (auto device : devices) {
      // Get device properties
      ze_device_properties_t deviceProperties;
      CALL_ZE_AND_CHECK(zeDeviceGetProperties, device, &deviceProperties);
      // Print device name
      llvm::outs() << deviceProperties.name << '\n';
    }
  }

  return 0;
}

#endif // HAVE_LEVEL_ZERO_HEADERS
