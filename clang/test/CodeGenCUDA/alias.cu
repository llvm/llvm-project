// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -fcuda-is-device -triple nvptx-nvidia-cuda -emit-llvm \
// RUN:   -o - %s | FileCheck %s
// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn -emit-llvm \
// RUN:   -o - %s | FileCheck %s

#include "Inputs/cuda.h"

// Check that we don't generate an alias from "foo" to the mangled name for
// ns::foo() -- nvptx doesn't support aliases.

namespace ns {
extern "C" {
// CHECK-NOT: @foo = internal alias
__device__ __attribute__((used)) static int foo() { return 0; }
}
}
