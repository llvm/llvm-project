// RUN: echo "GPU binary would be here" > %t
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -foffload-include-binary %t -o - \
// RUN:   | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -foffload-include-binary %t -o - -x hip \
// RUN:   | FileCheck %s

#include "Inputs/cuda.h"

__global__ void kernel() {}

struct Launcher {
  Launcher() { kernel<<<1, 1>>>(); }
} launcher;

// The constructor that registers the device binary precedes the static
// initializer that launches the kernel.
// CHECK: @llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__{{cuda|hip}}_module_ctor, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_{{[^,]+}}, ptr null }]
