// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target
// REQUIRES: libgcc
// UNSUPPORTED: system-windows

// RUN: %clang -c -### -target x86_64-unknown-linux-gnu --cuda-gpu-arch=gfx900 \
// RUN:   -std=c++11 --rocm-path=%S/Inputs/rocm -nogpulib %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=COMMON,CLANG,HIP %s

// RUN: %clang -c -### -target x86_64-unknown-linux-gnu --cuda-gpu-arch=gfx900 \
// RUN:   -std=c++11 --rocm-path=%S/Inputs/rocm -nobuiltininc -nogpulib %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=COMMON,NOCLANG,HIP %s

// RUN: %clang -c -### -target x86_64-unknown-linux-gnu --cuda-gpu-arch=gfx900 \
// RUN:   -std=c++11 --rocm-path=%S/Inputs/rocm -nogpuinc -nogpulib %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=COMMON,CLANG,NOHIP %s

// COMMON-LABEL: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CLANG-SAME: "-internal-isystem" "{{.*}}clang/{{.*}}/include/cuda_wrappers"
// NOCLANG-NOT: "{{.*}}clang/{{.*}}/include/cuda_wrappers"
// CLANG-SAME: "-include" "__clang_hip_runtime_wrapper.h"
// HIP-SAME: "-internal-isystem" "{{.*}}Inputs/rocm/include"
// NOHIP-NOT: "{{.*}}Inputs/rocm/include"
// skip check of standard C++ include path
// CLANG-SAME: "-internal-isystem" "{{.*}}clang/{{.*}}/include"
// NOCLANG-NOT: "{{.*}}clang/{{.*}}/include"

// COMMON-LABEL: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CLANG-SAME: "-internal-isystem" "{{.*}}clang/{{.*}}/include/cuda_wrappers"
// NOCLANG-NOT: "{{.*}}clang/{{.*}}/include/cuda_wrappers"
// CLANG-SAME: "-include" "__clang_hip_runtime_wrapper.h"
// HIP-SAME: "-internal-isystem" "{{.*}}Inputs/rocm/include"
// NOHIP-NOT: "{{.*}}Inputs/rocm/include"
// skip check of standard C++ include path
// CLANG-SAME: "-internal-isystem" "{{.*}}clang/{{.*}}/include"
// NOCLANG-NOT: "{{.*}}clang/{{.*}}/include"
