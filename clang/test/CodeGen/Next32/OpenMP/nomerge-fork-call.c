// RUN: %clang_cc1 -triple x86_64 -O3 -fopenmp -emit-llvm %s -o - | FileCheck %s

// CHECK: @__kmpc_fork_call(ptr, i32, ptr, ...) local_unnamed_addr #[[ATTR:[0-9]+]]
// CHECK: attributes #[[ATTR]] = {{{.*}}nomerge{{.*}}}

void foo(int *a, int size) {
#pragma omp parallel for
  for (int i = 0; i < size; ++i)
    a[i] += 5;
}
