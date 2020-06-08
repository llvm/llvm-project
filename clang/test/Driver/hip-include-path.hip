// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -c -v -target x86_64-unknown-linux-gnu --cuda-gpu-arch=gfx900 \
// RUN:   -std=c++11 --rocm-path=%S/Inputs/rocm -nogpulib %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=COMMON,WRAP,HIP %s

// RUN: %clang -c -v -target x86_64-unknown-linux-gnu --cuda-gpu-arch=gfx900 \
// RUN:   -std=c++11 --rocm-path=%S/Inputs/rocm -nobuiltininc -nogpulib %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=COMMON,NOWRAP,HIP %s

// RUN: %clang -c -v -target x86_64-unknown-linux-gnu --cuda-gpu-arch=gfx900 \
// RUN:   -std=c++11 --rocm-path=%S/Inputs/rocm -nogpuinc -nogpulib %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=COMMON,WRAP,NOHIP %s

// COMMON-LABEL: clang{{.*}} -cc1 -triple amdgcn-amd-amdhsa
// WRAP: clang/{{.*}}/include/cuda_wrappers
// NOWRAP-NOT: clang/{{.*}}/include/cuda_wrappers
// HIP: {{.*}}Inputs/rocm/include
// NOHIP-NOT: {{.*}}Inputs/rocm/include
// skip check of standard C++ include path
// COMMON: clang/{{.*}}/include

// COMMON-LABEL: clang{{.*}} -cc1 -triple x86_64
// WRAP: clang/{{.*}}/include/cuda_wrappers
// NOWRAP-NOT: clang/{{.*}}/include/cuda_wrappers
// HIP: {{.*}}Inputs/rocm/include
// NOHIP-NOT: {{.*}}Inputs/rocm/include
// skip check of standard C++ include path
// COMMON: clang/{{.*}}/include
