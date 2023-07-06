// REQUIRES: nvptx-registered-target
// REQUIRES: amdgpu-registered-target

// RUN:   %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx1030 -nogpulib \
// RUN:     -nogpuinc %s 2>&1 | FileCheck %s --check-prefix=CHECK-HEADERS-DISABLED
// RUN:   %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx1030 -nogpulib \
// RUN:     -nostdinc %s 2>&1 | FileCheck %s --check-prefix=CHECK-HEADERS-DISABLED
// RUN:   %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx1030 -nogpulib \
// RUN:     -nobuiltininc %s 2>&1 | FileCheck %s --check-prefix=CHECK-HEADERS-DISABLED
// CHECK-HEADERS-DISABLED-NOT: "-cc1"{{.*}}"-c-isystem" "{{.*}}include{{.*}}gpu-none-llvm"
