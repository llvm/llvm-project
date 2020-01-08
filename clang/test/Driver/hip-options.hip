// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -### -x hip --gpu-max-threads-per-block=1024 %s 2>&1 | FileCheck %s

// Check that there are commands for both host- and device-side compilations.
//
// CHECK: clang{{.*}}" "-cc1" {{.*}} "-fcuda-is-device"
// CHECK-SAME: "--gpu-max-threads-per-block=1024"
