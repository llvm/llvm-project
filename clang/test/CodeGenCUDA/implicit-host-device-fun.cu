// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu \
// RUN:   -foffload-implicit-host-device-templates \
// RUN:   -emit-llvm -o - -x hip %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=COMM,HOST %s 
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:   -target-cpu gfx1100 \
// RUN:   -foffload-implicit-host-device-templates \
// RUN:   -emit-llvm -o - -x hip %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=COMM,DEV %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:   -target-cpu gfx1100 \
// RUN:   -foffload-implicit-host-device-templates \
// RUN:   -emit-llvm -o - -x hip %s 2>&1 | \
// RUN:   FileCheck -check-prefixes=DEV-NEG %s

#include "Inputs/cuda.h"

// Implicit host device template not overloaded by device template.
// Used by both device and host function.
// Emitted on both host and device.

// COMM-LABEL: define {{.*}}@_Z20template_no_overloadIiET_S0_(
// COMM:  ret i32 1
template<typename T>
T template_no_overload(T x) {
  return 1;
}

// Implicit host device template overloaded by device template.
// Used by both device and host function.
// Implicit host device template emitted on host.
// Device template emitted on device.

// COMM-LABEL: define {{.*}}@_Z22template_with_overloadIiET_S0_(
// HOST:  ret i32 2
// DEV:  ret i32 3
template<typename T>
T template_with_overload(T x) {
  return 2;
}

template<typename T>
__device__ T template_with_overload(T x) {
  return 3;
}

// Implicit host device template used by host function only.
// Emitted on host only.
// HOST-LABEL: define {{.*}}@_Z21template_used_by_hostIiET_S0_(
// DEV-NEG-NOT: define {{.*}}@_Z21template_used_by_hostIiET_S0_(
// HOST:  ret i32 10
template<typename T>
T template_used_by_host(T x) {
  return 10;
}

// Implicit host device template indirectly used by host function only.
// Emitted on host only.
// HOST-LABEL: define {{.*}}@_Z32template_indirectly_used_by_hostIiET_S0_(
// DEV-NEG-NOT: define {{.*}}@_Z32template_indirectly_used_by_hostIiET_S0_(
// HOST:  ret i32 11
template<typename T>
T template_indirectly_used_by_host(T x) {
  return 11;
}

template<typename T>
T template_in_middle_by_host(T x) {
  template_indirectly_used_by_host(x);
  return 12;
}

// Implicit host device template indirectly used by device function only.
// Emitted on device.
// DEVICE-LABEL: define {{.*}}@_Z34template_indirectly_used_by_deviceIiET_S0_(
// DEVICE:  ret i32 21
template<typename T>
T template_indirectly_used_by_device(T x) {
  return 21;
}

template<typename T>
T template_in_middle_by_device(T x) {
  template_indirectly_used_by_device(x);
  return 22;
}

// Implicit host device template indirectly used by host device function only.
// Emitted on host and device.
// COMMON-LABEL: define {{.*}}@_Z39template_indirectly_used_by_host_deviceIiET_S0_(
// COMMON:  ret i32 31
template<typename T>
T template_indirectly_used_by_host_device(T x) {
  return 31;
}

template<typename T>
T template_in_middle_by_host_device(T x) {
  template_indirectly_used_by_host_device(x);
  return 32;
}

void host_fun() {
  template_no_overload(0);
  template_with_overload(0);
  template_used_by_host(0);
  template_in_middle_by_host(0);
}

__device__ void device_fun() {
  template_no_overload(0);
  template_with_overload(0);
  template_in_middle_by_device(0);
}

__host__ __device__ void host_device_fun() {
  template_in_middle_by_host_device(0);
}
