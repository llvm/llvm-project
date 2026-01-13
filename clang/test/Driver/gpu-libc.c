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


// RUN:   %clang -### -fopenmp=libomp --target=x86_64-unknown-linux-gnu \
// RUN:     --offload-arch=gfx908 --rocm-path=%S/Inputs/rocm --sysroot=%S/Inputs/basic_gpu_tree \
// RUN:     -ccc-install-dir %S/Inputs/basic_gpu_tree/bin %s 2>&1 | FileCheck %s --check-prefix=OPENMP-AMDGPU
// OPENMP-AMDGPU: clang-linker-wrapper{{.*}}"--device-linker=amdgcn-amd-amdhsa=-lc"
// RUN:   %clang -### -fopenmp=libomp --target=x86_64-unknown-linux-gnu -foffload-lto \
// RUN:     --offload-arch=sm_52 --cuda-path=%S/Inputs/CUDA_111/usr/local/cuda --sysroot=%S/Inputs/basic_gpu_tree \
// RUN:     -ccc-install-dir %S/Inputs/basic_gpu_tree/bin %s 2>&1 | FileCheck %s --check-prefix=OPENMP-NVPTX
// OPENMP-NVPTX: clang-linker-wrapper{{.*}}"--device-linker=nvptx64-nvidia-cuda=-lc"
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu --offload-arch=gfx908 \
// RUN:     --offload-new-driver --rocm-path=%S/Inputs/rocm --sysroot=%S/Inputs/basic_gpu_tree \
// RUN:     -ccc-install-dir %S/Inputs/basic_gpu_tree/bin -x hip %s 2>&1 | FileCheck %s --check-prefix=HIP
// HIP-NOT: "--device-linker=amdgcn-amd-amdhsa=-lc"
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -fgpu-rdc --offload-arch=sm_52 \
// RUN:     --cuda-path=%S/Inputs/CUDA_111/usr/local/cuda --sysroot=%S/Inputs/basic_gpu_tree \
// RUN:     -ccc-install-dir %S/Inputs/basic_gpu_tree/bin -x cuda %s 2>&1 | FileCheck %s --check-prefix=CUDA
// CUDA-NOT: "--device-linker=nvptx64-nvidia-cuda=-lc"
