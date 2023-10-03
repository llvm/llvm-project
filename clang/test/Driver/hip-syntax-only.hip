// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -### -nogpulib -nogpuinc --target=x86_64 -fsyntax-only %s 2>&1 | FileCheck %s

// Check that there are commands for both host- and device-side compilations.
//
// CHECK-DAG: "-cc1" {{.*}} "-fcuda-is-device"
// CHECK-DAG: "-cc1" "-triple" "x86_64"
// CHECK-NOT: clang-offload-bundler"
