// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
//
// RUN: %clang -v --target=i386-unknown-linux \
// RUN:   --sysroot=/tmp/no-cuda-there 2>&1 | FileCheck %s -check-prefix NOCUDA
// RUN: %clang -v --target=i386-unknown-linux \
// RUN:   --cuda-path=%S/Inputs/CUDA/usr/local/cuda 2>&1 | FileCheck %s

// CHECK: Found CUDA installation: {{.*}}/Inputs/CUDA/usr/local/cuda
// NOCUDA-NOT: Found CUDA installation:
