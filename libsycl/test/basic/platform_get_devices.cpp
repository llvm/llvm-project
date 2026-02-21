// REQUIRES: any-device
// RUN: %clangxx %sycl_options %s -o %t.out
// RUN: %t.out
//
// Tests platform::get_devices for each device type.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <iostream>

std::string BackendToString(sycl::backend Backend) {
  switch (Backend) {
  case sycl::backend::opencl:
    return "opencl";
  case sycl::backend::level_zero:
    return "level_zero";
  case sycl::backend::cuda:
    return "cuda";
  case sycl::backend::hip:
    return "hip";
  default:
    return "unknown";
  }
}

std::string DeviceTypeToString(sycl::info::device_type DevType) {
  switch (DevType) {
  case sycl::info::device_type::all:
    return "device_type::all";
  case sycl::info::device_type::cpu:
    return "device_type::cpu";
  case sycl::info::device_type::gpu:
    return "device_type::gpu";
  case sycl::info::device_type::accelerator:
    return "device_type::accelerator";
  case sycl::info::device_type::custom:
    return "device_type::custom";
  case sycl::info::device_type::automatic:
    return "device_type::automatic";
  case sycl::info::device_type::host:
    return "device_type::host";
  default:
    return "unknown";
  }
}

std::string GenerateDeviceDescription(sycl::info::device_type DevType,
                                      const sycl::platform &Platform) {
  return std::string(DeviceTypeToString(DevType)) + " (" +
         BackendToString(Platform.get_backend()) + ")";
}

template <typename T1, typename T2>
int Check(const T1 &LHS, const T2 &RHS, std::string TestName) {
  if (LHS == RHS)
    return 0;

  std::cerr << "Failed check " << LHS << " != " << RHS << ": " << TestName
            << std::endl;
  return 1;
}

int CheckDeviceType(const sycl::platform &P, sycl::info::device_type DevType,
                    std::vector<sycl::device> &AllDevices) {
  // This check verifies data of device with specific device_type and if it is
  // correctly chosen among all devices (device_type::all).
  // Though there is no point to check device_type::all here.
  assert(DevType != sycl::info::device_type::all);
  int Failures = 0;

  std::vector<sycl::device> Devices = P.get_devices(DevType);

  if (DevType == sycl::info::device_type::automatic) {
    if (AllDevices.empty()) {
      Failures += Check(Devices.size(), 0,
                        "No devices reported for device_type::all query, but "
                        "device_type::automatic returns a device.");
    } else {
      Failures += Check(Devices.size(), 1,
                        "Number of devices for device_type::automatic query.");
      if (Devices.size())
        Failures += Check(
            std::count(AllDevices.begin(), AllDevices.end(), Devices[0]), 1,
            "Device is in the set of device_type::all devices in the "
            "platform.");
    }
    return Failures;
  }

  // Count devices with the type.
  size_t DevCount = 0;
  for (sycl::device Device : Devices)
    DevCount += (Device.get_info<sycl::info::device::device_type>() == DevType);

  Failures += Check(Devices.size(), DevCount,
                    "Unexpected number of devices for " +
                        GenerateDeviceDescription(DevType, P));

  Failures +=
      Check(std::all_of(Devices.begin(), Devices.end(),
                        [&](const auto &Dev) {
                          return std::count(AllDevices.begin(),
                                            AllDevices.end(), Dev) == 1;
                        }),
            true,
            "Not all devices for " + GenerateDeviceDescription(DevType, P) +
                " appear in the list of all devices");

  return Failures;
}

int main() {
  int Failures = 0;
  for (sycl::platform P : sycl::platform::get_platforms()) {
    std::vector<sycl::device> Devices = P.get_devices();

    for (sycl::info::device_type DevType :
         {sycl::info::device_type::cpu, sycl::info::device_type::gpu,
          sycl::info::device_type::accelerator, sycl::info::device_type::custom,
          sycl::info::device_type::automatic, sycl::info::device_type::host})
      Failures += CheckDeviceType(P, DevType, Devices);
  }
  return Failures;
}
