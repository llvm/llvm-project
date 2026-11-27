//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out
//
// Queries all supported device information descriptors for every device and
// prints the results.

#include <sycl/sycl.hpp>

#include <cassert>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

using namespace sycl;

std::ostream &operator<<(std::ostream &OS, info::device_type DevType) {
  switch (DevType) {
  case info::device_type::cpu:
    return OS << "cpu";
  case info::device_type::gpu:
    return OS << "gpu";
  case info::device_type::accelerator:
    return OS << "accelerator";
  case info::device_type::custom:
    return OS << "custom";
  case info::device_type::automatic:
    return OS << "automatic";
  case info::device_type::host:
    return OS << "host";
  case info::device_type::all:
    return OS << "all";
  }
  return OS << "unknown";
}

std::ostream &operator<<(std::ostream &OS, info::fp_config Config) {
  switch (Config) {
  case info::fp_config::denorm:
    return OS << "denorm";
  case info::fp_config::inf_nan:
    return OS << "inf_nan";
  case info::fp_config::round_to_nearest:
    return OS << "round_to_nearest";
  case info::fp_config::round_to_zero:
    return OS << "round_to_zero";
  case info::fp_config::round_to_inf:
    return OS << "round_to_inf";
  case info::fp_config::fma:
    return OS << "fma";
  case info::fp_config::correctly_rounded_divide_sqrt:
    return OS << "correctly_rounded_divide_sqrt";
  case info::fp_config::soft_float:
    return OS << "soft_float";
  }
  return OS << "unknown";
}

template <int Dimensions>
std::ostream &operator<<(std::ostream &OS, const range<Dimensions> &Range) {
  OS << "{";
  for (int I = 0; I < Dimensions; ++I)
    OS << (I ? ", " : "") << Range[I];
  return OS << "}";
}

template <typename T>
std::ostream &operator<<(std::ostream &OS, const std::vector<T> &Values) {
  OS << "{";
  for (std::size_t I = 0; I < Values.size(); ++I)
    OS << (I ? ", " : "") << Values[I];
  return OS << "}";
}

// Prints the platform name rather than the platform object.
std::ostream &operator<<(std::ostream &OS, const platform &Platform) {
  return OS << Platform.get_info<info::platform::name>();
}

template <typename Desc>
typename Desc::return_type printInfo(const device &Dev, const char *Name) {
  auto Value = Dev.get_info<Desc>();
  std::cout << "  " << Name << ": " << Value << std::endl;
  return Value;
}

#define PRINT_INFO(Desc) printInfo<info::device::Desc>(Dev, #Desc)

void printDeviceInfo(const device &Dev) {
  PRINT_INFO(device_type);
  PRINT_INFO(vendor_id);
  auto ComputeUnits = PRINT_INFO(max_compute_units);
  auto WIDims = PRINT_INFO(max_work_item_dimensions);
  auto WISizes1 = PRINT_INFO(max_work_item_sizes<1>);
  auto WISizes2 = PRINT_INFO(max_work_item_sizes<2>);
  auto WISizes3 = PRINT_INFO(max_work_item_sizes<3>);
  auto WGSize = PRINT_INFO(max_work_group_size);
  PRINT_INFO(preferred_vector_width_char);
  PRINT_INFO(preferred_vector_width_short);
  PRINT_INFO(preferred_vector_width_int);
  PRINT_INFO(preferred_vector_width_long);
  PRINT_INFO(preferred_vector_width_float);
  PRINT_INFO(preferred_vector_width_double);
  PRINT_INFO(preferred_vector_width_half);
  PRINT_INFO(native_vector_width_char);
  PRINT_INFO(native_vector_width_short);
  PRINT_INFO(native_vector_width_int);
  PRINT_INFO(native_vector_width_long);
  PRINT_INFO(native_vector_width_float);
  PRINT_INFO(native_vector_width_double);
  PRINT_INFO(native_vector_width_half);
  PRINT_INFO(max_clock_frequency);
  auto AddressBits = PRINT_INFO(address_bits);
  auto MaxAlloc = PRINT_INFO(max_mem_alloc_size);
  PRINT_INFO(half_fp_config);
  PRINT_INFO(single_fp_config);
  PRINT_INFO(double_fp_config);
  auto GlobalMem = PRINT_INFO(global_mem_size);
  PRINT_INFO(local_mem_size);
  auto Platform = PRINT_INFO(platform);
  PRINT_INFO(name);
  PRINT_INFO(vendor);
  PRINT_INFO(driver_version);

  // Basic sanity checks of the reported values.
  assert(ComputeUnits > 0);
  assert(WIDims == 3);
  assert(WGSize > 0);
  assert(WISizes1[0] > 0);
  assert(WISizes2[0] > 0 && WISizes2[1] > 0);
  assert(WISizes3[0] > 0 && WISizes3[1] > 0 && WISizes3[2] > 0);
  // Lower dimensional queries return the fastest varying dimensions.
  assert(WISizes1[0] == WISizes3[2]);
  assert(WISizes2[0] == WISizes3[1] && WISizes2[1] == WISizes3[2]);
  assert(AddressBits == 32 || AddressBits == 64);
  assert(MaxAlloc > 0 && MaxAlloc <= GlobalMem);
  assert(Platform == Dev.get_platform());
}

#undef PRINT_INFO

int main() {
  for (const auto &Platform : platform::get_platforms()) {
    for (const auto &Dev : Platform.get_devices()) {
      std::cout << "Device: " << Dev.get_info<info::device::name>()
                << std::endl;
      printDeviceInfo(Dev);
    }
  }
  return 0;
}
