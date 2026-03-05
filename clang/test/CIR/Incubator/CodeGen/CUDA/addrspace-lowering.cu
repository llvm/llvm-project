#include "../Inputs/cuda.h"

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-DEVICE --input-file=%t.ll %s


__shared__ int a;

// LLVM-DEVICE: @a = addrspace(3) {{.*}}

__device__ int b;

// LLVM-DEVICE: @b = addrspace(1) {{.*}}

__constant__ int c;

// LLVM-DEVICE: @c = addrspace(4) {{.*}}
