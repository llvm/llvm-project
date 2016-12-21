// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// RUN: %clang -### --target=i386-unknown-linux \
// RUN:   --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda \
// RUN:   --ptxas-path=/some/path/to/ptxas %s 2>&1 \
// RUN: | FileCheck %s

// CHECK-NOT: "ptxas"
// CHECK: "/some/path/to/ptxas"
// CHECK-SAME: "--gpu-name" "sm_20"
