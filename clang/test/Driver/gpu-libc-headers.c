// RUN:   %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx90a --sysroot=%S/Inputs/basic_gpu_tree \
// RUN:     -ccc-install-dir %S/Inputs/basic_gpu_tree/bin -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-HEADERS-AMDGPU
// RUN:   %clang -### --target=nvptx64-nvidia-cuda -march=sm_89 --sysroot=%S/Inputs/basic_gpu_tree \
// RUN:     -ccc-install-dir %S/Inputs/basic_gpu_tree/bin -nogpulib %s 2>&1 | FileCheck %s --check-prefix=CHECK-HEADERS-NVPTX
// CHECK-HEADERS-AMDGPU: "-cc1"{{.*}}"-isysroot"{{.*}}"-internal-isystem" "{{.*}}include{{.*}}amdgcn-amd-amdhsa"
// CHECK-HEADERS-NVPTX: "-cc1"{{.*}}"-isysroot"{{.*}}"-internal-isystem" "{{.*}}include{{.*}}nvptx64-nvidia-cuda"

// RUN:   %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx1030 -nogpulib \
// RUN:     -nogpuinc %s 2>&1 | FileCheck %s --check-prefix=CHECK-HEADERS-DISABLED
// RUN:   %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx1030 -nogpulib \
// RUN:     -nostdinc %s 2>&1 | FileCheck %s --check-prefix=CHECK-HEADERS-DISABLED
// RUN:   %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx1030 -nogpulib \
// RUN:     -nobuiltininc %s 2>&1 | FileCheck %s --check-prefix=CHECK-HEADERS-DISABLED
// CHECK-HEADERS-DISABLED-NOT: "-cc1"{{.*}}"-internal-isystem" "{{.*}}include{{.*}}gpu-none-llvm"
