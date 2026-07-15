// REQUIRES: amdgpu-registered-target, x86-registered-target

// Check that -mllvm options are not duplicated on the device -cc1 command line.

// RUN: %clang -### -target x86_64-pc-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -nogpulib \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 \
// RUN:   -mllvm -amdgpu-dump-hsa-metadata \
// RUN:   %s 2>&1 | FileCheck %s

// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-pc-linux-gnu"
// CHECK-SAME: "-target-cpu" "gfx906"
// CHECK-SAME: "-fopenmp"
// CHECK-SAME: "-mllvm" "-amdgpu-dump-hsa-metadata"
// CHECK-NOT:  "-mllvm" "-amdgpu-dump-hsa-metadata" "-mllvm" "-amdgpu-dump-hsa-metadata"
// CHECK-SAME: "-fopenmp-is-target-device"
