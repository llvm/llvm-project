// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -### -target x86_64-pc-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -nogpulib \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 \
// RUN:   -mllvm -amdgpu-dump-hsa-metadata \
// RUN:   %s 2>&1 | FileCheck %s

// RUN: %clang -### -target x86_64-pc-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -nogpulib \
// RUN:   -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 \
// RUN:   -mllvm -amdgpu-dump-hsa-metadata \
// RUN:   %s 2>&1 | FileCheck --check-prefix=DUP %s

// CHECK: [[CLANG:".*clang.*"]] "-cc1" "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-aux-triple" "x86_64-pc-linux-gnu"
// CHECK-SAME: "-emit-llvm-bc" {{.*}} "-target-cpu" "gfx906"
// CHECK-SAME: "-fopenmp"
// CHECK-SAME:  "-mllvm" "-amdgpu-dump-hsa-metadata"
// DUP-NOT:  "-mllvm" "-amdgpu-dump-hsa-metadata" "-mllvm" "-amdgpu-dump-hsa-metadata"
// CHECK-SAME: "-fopenmp-is-device"

// CHECK: [[OPT:".*llc.*"]] {{".*-gfx906-optimized.*bc"}} "-mtriple=amdgcn-amd-amdhsa"
// CHECK-SAME: "-mcpu=gfx906"
// CHECK-SAME: "-amdgpu-dump-hsa-metadata"
// DUP-NOT: "-amdgpu-dump-hsa-metadata" "-amdgpu-dump-hsa-metadata"
