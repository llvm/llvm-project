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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include <cstdio>

#define ZE_MAX_DEVICE_NAME 256
#define ZE_MAX_DEVICE_UUID_SIZE 16

using ze_driver_handle_t = void *;
using ze_device_handle_t = void *;

enum ze_result_t {
  ZE_RESULT_SUCCESS = 0,
  ZE_RESULT_ERROR_UNKNOWN = 0x7ffffffe
};

enum ze_structure_type_t {
  ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC = 0x00020021,
  ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES = 0x3,
  ZE_STRUCTURE_TYPE_FORCE_UINT32 = 0x7fffffff
};

enum ze_init_driver_type_flags_t { ZE_INIT_DRIVER_TYPE_FLAG_GPU = 1 };

using ze_device_type_t = uint32_t;
using ze_device_property_flags_t = uint32_t;

struct ze_init_driver_type_desc_t {
  ze_structure_type_t stype;
  const void *pNext;
  ze_init_driver_type_flags_t flags;
};

struct ze_device_uuid_t {
  uint8_t id[ZE_MAX_DEVICE_UUID_SIZE];
};

struct ze_device_properties_t {
  ze_structure_type_t stype;
  void *pNext;
  ze_device_type_t type;
  uint32_t vendorId;
  uint32_t deviceId;
  ze_device_property_flags_t flags;
  uint32_t subdeviceId;
  uint32_t coreClockRate;
  uint64_t maxMemAllocSize;
  uint32_t maxHardwareContexts;
  uint32_t maxCommandQueuePriority;
  uint32_t numThreadsPerEU;
  uint32_t physicalEUSimdWidth;
  uint32_t numEUsPerSubslice;
  uint32_t numSubslicesPerSlice;
  uint32_t numSlices;
  uint64_t timerResolution;
  uint32_t timestampValidBits;
  uint32_t kernelTimestampValidBits;
  ze_device_uuid_t uuid;
  char name[ZE_MAX_DEVICE_NAME];
};

ze_result_t zeInitDrivers(uint32_t *pCount, ze_driver_handle_t *phDrivers,
                          ze_init_driver_type_desc_t *desc);
ze_result_t zeDeviceGet(ze_driver_handle_t hDriver, uint32_t *pCount,
                        void *phDevices);
ze_result_t zeDeviceGetProperties(void *hDevice, void *pProperties);

using namespace llvm;
extern cl::opt<bool> Verbose;

#define DEFINE_WRAPPER(NAME)                                                   \
  using NAME##_ty = decltype(NAME);                                            \
  void *NAME##Ptr = nullptr;                                                   \
  template <class... Ts> ze_result_t NAME##Wrapper(Ts... args) {               \
    if (!NAME##Ptr) {                                                          \
      return ZE_RESULT_ERROR_UNKNOWN;                                          \
    }                                                                          \
    return reinterpret_cast<NAME##_ty *>(NAME##Ptr)(args...);                  \
  }

DEFINE_WRAPPER(zeInitDrivers)
DEFINE_WRAPPER(zeDeviceGet)
DEFINE_WRAPPER(zeDeviceGetProperties)

static bool loadLevelZero() {
  constexpr const char *L0Library = "libze_loader.so";
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
    const char *Name;
    void **FuncPtr;
  } Wrappers[] = {
      {"zeInitDrivers", &zeInitDriversPtr},
      {"zeDeviceGet", &zeDeviceGetPtr},
      {"zeDeviceGetProperties", &zeDeviceGetPropertiesPtr},
  };

  for (auto Entry : Wrappers) {
    void *P = DynlibHandle->getAddressOfSymbol(Entry.Name);
    if (P == nullptr) {
      if (Verbose)
        llvm::errs() << "Unable to find '" << Entry.Name << "' in '"
                     << L0Library << "'\n";
      return false;
    }
    *(Entry.FuncPtr) = P;
  }

  return true;
}

#define CALL_ZE_AND_CHECK(Fn, ...)                                             \
  do {                                                                         \
    ze_result_t Rc = Fn##Wrapper(__VA_ARGS__);                                 \
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

  ze_init_driver_type_desc_t DriverType = {};
  DriverType.stype = ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC;
  DriverType.flags = ZE_INIT_DRIVER_TYPE_FLAG_GPU;
  DriverType.pNext = nullptr;
  uint32_t DriverCount{0};

  // Initialize and find all drivers.
  CALL_ZE_AND_CHECK(zeInitDrivers, &DriverCount, nullptr, &DriverType);

  llvm::SmallVector<ze_driver_handle_t> Drivers(DriverCount);
  CALL_ZE_AND_CHECK(zeInitDrivers, &DriverCount, Drivers.data(), &DriverType);

  for (auto Driver : Drivers) {
    // Discover all the devices for a given driver.
    uint32_t DeviceCount = 0;
    CALL_ZE_AND_CHECK(zeDeviceGet, Driver, &DeviceCount, nullptr);

    llvm::SmallVector<ze_device_handle_t> Devices(DeviceCount);
    CALL_ZE_AND_CHECK(zeDeviceGet, Driver, &DeviceCount, Devices.data());

    for (auto Device : Devices) {
      ze_device_properties_t DeviceProperties = {};
      DeviceProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
      DeviceProperties.pNext = nullptr;
      CALL_ZE_AND_CHECK(zeDeviceGetProperties, Device, &DeviceProperties);
      llvm::outs() << DeviceProperties.name << '\n';
    }
  }

  return 0;
}
