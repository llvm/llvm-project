// UNSUPPORTED: system-windows

// RUN: %clang -c -### --target=x86_64-unknown-linux-gnu --cuda-gpu-arch=gfx900 \
// RUN:   -std=c++11 --rocm-path=%S/Inputs/rocm -nogpulib %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=COMMON,CLANG,HIP -DRESOURCE_DIR=%clang-resource-dir %s

// RUN: %clang -c -### --target=x86_64-unknown-linux-gnu --cuda-gpu-arch=gfx900 \
// RUN:   -std=c++11 --rocm-path=%S/Inputs/rocm -nobuiltininc -nogpulib %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=COMMON,NOCLANG,HIP -DRESOURCE_DIR=%clang-resource-dir %s

// RUN: %clang -c -### --target=x86_64-unknown-linux-gnu --cuda-gpu-arch=gfx900 \
// RUN:   -std=c++11 --rocm-path=%S/Inputs/rocm -nogpuinc -nogpulib %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=COMMON,CLANG,NOHIP -DRESOURCE_DIR=%clang-resource-dir %s

// RUN: %clang -c -### --target=x86_64-unknown-linux-gnu --cuda-gpu-arch=gfx900 \
// RUN:   -std=c++11 --rocm-path=%S/Inputs/rocm --no-offload-inc -nogpulib --offload-inc %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=COMMON,CLANG,HIP -DRESOURCE_DIR=%clang-resource-dir %s

// RUN: %clang -c -### --target=x86_64-unknown-linux-gnu --cuda-gpu-arch=gfx900 \
// RUN:   -std=c++11 --rocm-path=%S/Inputs/rocm -nogpulib -nostdinc++ %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=COMMON,CLANG,HIP -DRESOURCE_DIR=%clang-resource-dir %s

// RUN: %clang -c -### --target=x86_64-unknown-linux-gnu --cuda-gpu-arch=gfx900 \
// RUN:   -std=c++11 --rocm-path=%S/Inputs/rocm-invalid -nogpulib %s 2>&1 \
// RUN:   | FileCheck -check-prefix=NOLIBHIPCXX %s

// NOLIBHIPCXX: "-internal-isystem" "{{[^"]*}}Inputs/rocm-invalid/include"
// NOLIBHIPCXX-NOT: "{{.*}}Inputs/rocm-invalid/include/libhipcxx"

// COMMON-LABEL: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CLANG-SAME: "-internal-isystem" "[[RESOURCE_DIR]]/include/cuda_wrappers"
// NOCLANG-NOT: "[[RESOURCE_DIR]]/include/cuda_wrappers"
// HIP-SAME: "-include" "__clang_hip_runtime_wrapper.h"
// NOHIP-NOT: "-include" "__clang_hip_runtime_wrapper.h"
// skip check of standard C++ include path
// ROCm toolkit includes come after C++ stdlib but before system includes
// HIP-SAME: "-internal-isystem" "{{[^"]*}}Inputs/rocm/include"
// HIP-SAME: "-internal-isystem" "{{[^"]*}}Inputs/rocm/include/libhipcxx"
// NOHIP-NOT: "{{.*}}Inputs/rocm/include"
// CLANG-SAME: "-internal-isystem" "[[RESOURCE_DIR]]/include"
// NOCLANG-NOT: "[[RESOURCE_DIR]]/include"

// COMMON-LABEL: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CLANG-SAME: "-internal-isystem" "[[RESOURCE_DIR]]/include/cuda_wrappers"
// NOCLANG-NOT: "[[RESOURCE_DIR]]/include/cuda_wrappers"
// HIP-SAME: "-include" "__clang_hip_runtime_wrapper.h"
// NOHIP-NOT: "-include" "__clang_hip_runtime_wrapper.h"
// skip check of standard C++ include path
// ROCm toolkit includes come after C++ stdlib but before system includes
// HIP-SAME: "-internal-isystem" "{{[^"]*}}Inputs/rocm/include"
// HIP-SAME: "-internal-isystem" "{{[^"]*}}Inputs/rocm/include/libhipcxx"
// NOHIP-NOT: "{{.*}}Inputs/rocm/include"
// CLANG-SAME: "-internal-isystem" "[[RESOURCE_DIR]]/include"
// NOCLANG-NOT: "[[RESOURCE_DIR]]/include"

// RUN: %clang -c -### --target=x86_64-unknown-linux-gnu --cuda-gpu-arch=gfx900 \
// RUN:   -std=c++11 --rocm-path=%S/Inputs/rocm -nogpulib %s 2>&1 \
// RUN:   --hip-version=3.5 | FileCheck -check-prefixes=ROCM35 -DRESOURCE_DIR=%clang-resource-dir %s

// ROCM35-LABEL: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// ROCM35-NOT: "[[RESOURCE_DIR]]/include/cuda_wrappers"
// ROCM35-SAME: "-internal-isystem" "[[RESOURCE_DIR]]"
// ROCM35-NOT: "-include" "__clang_hip_runtime_wrapper.h"
// skip check of standard C++ include path
// ROCm toolkit includes come after C++ stdlib but before system includes
// ROCM35-SAME: "-internal-isystem" "{{[^"]*}}Inputs/rocm/include"
// ROCM35-SAME: "-internal-isystem" "{{[^"]*}}Inputs/rocm/include/libhipcxx"
// ROCM35-SAME: "-internal-isystem" "[[RESOURCE_DIR]]/include"

// The device compilation must pick up the host toolchain's system include paths
// so host C++ library headers are usable.
// RUN: %clang -c -### --target=x86_64-unknown-linux-gnu --cuda-gpu-arch=gfx900 \
// RUN:   -std=c++11 --rocm-path=%S/Inputs/rocm -nogpulib \
// RUN:   --sysroot=%S/Inputs/basic_linux_tree %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=HOSTINC %s

// HOSTINC-LABEL: "{{[^"]*}}clang{{[^"]*}}" "-cc1" "-triple" "amdgpu9.00-amd-amdhsa"
// HOSTINC: "-internal-externc-isystem" "{{[^"]*}}Inputs/basic_linux_tree/usr/include"

// Host-only compilation must still receive ROCm toolkit include paths via the
// late hook so that host code can use HIP runtime headers.
// RUN: %clang -c -### --target=x86_64-unknown-linux-gnu --cuda-gpu-arch=gfx900 \
// RUN:   -std=c++11 --rocm-path=%S/Inputs/rocm -nogpulib --cuda-host-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix=HOSTONLY %s

// HOSTONLY: "-internal-isystem" "{{[^"]*}}Inputs/rocm/include"
// HOSTONLY: "-internal-isystem" "{{[^"]*}}Inputs/rocm/include/libhipcxx"

// When using -foffload-via-llvm, vendor ROCm headers must not be added; LLVM's
// own offload headers are used instead.
// RUN: %clang -c -### --target=x86_64-unknown-linux-gnu --cuda-gpu-arch=gfx900 \
// RUN:   -std=c++11 --rocm-path=%S/Inputs/rocm -nogpulib -foffload-via-llvm %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LLVMOFFLOAD %s

// LLVMOFFLOAD-NOT: "-internal-isystem" "{{.*}}Inputs/rocm/include"
// LLVMOFFLOAD-NOT: "{{.*}}Inputs/rocm/include/libhipcxx"
// LLVMOFFLOAD-NOT: "-include" "__clang_hip_runtime_wrapper.h"
