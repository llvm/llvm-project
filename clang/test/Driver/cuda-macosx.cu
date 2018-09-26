// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
//
// RUN: %clang -v --target=i386-apple-macosx \
// RUN:   --sysroot=%S/Inputs/CUDA-macosx --cuda-path-ignore-env 2>&1 | FileCheck %s

// CHECK: Found CUDA installation: {{.*}}/Inputs/CUDA-macosx/usr/local/cuda
