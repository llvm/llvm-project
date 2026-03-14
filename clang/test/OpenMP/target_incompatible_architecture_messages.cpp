// RUN: not %clang_cc1 -x c++ -fopenmp -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -o - %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -x c++ -fopenmp -triple i386-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -o - %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -x c++ -fopenmp -triple x86_64-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -o - %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -x c++ -fopenmp -triple x86_64-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -o - %s 2>&1 | FileCheck %s
// CHECK: error: OpenMP target architecture '{{.+}}' pointer size is incompatible with host '{{.+}}'
#ifndef HEADER
#define HEADER

void test() {
#pragma omp target
  {}
}

#endif
