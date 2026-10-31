// The PTX version of the embedded fatbin must follow the pinned feature.
// RUN: rm -rf %t
// RUN: mkdir -p %t/lib
// RUN: %clang --target=%host-jit-triple -nostdlib -fPIC -shared \
// RUN:   %S/Inputs/cudart.c -o %t/lib/libcudart.so
// RUN: cat %s | clang-repl --cuda --cuda-path=%t -Xcc -nocudainc \
// RUN:   -Xcc -nocudalib -Xcc -include -Xcc %S/Inputs/cuda.h \
// RUN:   -Xcc -Xarch_device -Xcc --cuda-feature=+ptx80 \
// RUN:   -Xcc -Xclang -Xcc -emit-llvm | FileCheck %s

__global__ void kernel() {}
// CHECK: private constant {{.*}}.version 8.0{{.*}}.target sm_

%quit
