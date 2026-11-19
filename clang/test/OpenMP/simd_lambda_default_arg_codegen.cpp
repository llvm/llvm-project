// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fopenmp -emit-llvm -o - %s | FileCheck %s

int foo(auto x, int (*f)() = [] {
  int k = 0;
#pragma omp simd collapse(sizeof(x) / sizeof(x))
  for (int i = 0; i < 10; ++i)
    k++;
  return k;
}) {
  return f();
}
int bar() { return foo(1); }

// CHECK-LABEL: define internal noundef i32 @"{{.+}}clEv"(
// CHECK: store i32 %{{.+}}, ptr %{{.+}}, align 4, !llvm.access.group
// CHECK: br label %{{.+}}, !llvm.loop
// CHECK: !{!"llvm.loop.vectorize.enable"}
