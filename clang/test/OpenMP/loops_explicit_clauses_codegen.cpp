// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics


#ifndef HEADER
#define HEADER

#define N 10
int foo();
int bar();
int k;
// CHECK-LABEL: @main
int main(int argc, char **argv) {
  foo();
// CHECK: @{{.+}}foo
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK-NOT: @k
// CHECK: call void @__kmpc_for_static_fini(
// CHECK-NOT: @k
#pragma omp for private(k)
  for (k = 0; k < argc; k++)
    ;
  foo();
// CHECK: @{{.+}}foo
// CHECK: call void @__kmpc_for_static_init_8(
// CHECK-NOT: @k
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: store i32 %{{.+}}, ptr @k
#pragma omp for lastprivate(k) collapse(2)
  for (int i = 0; i < 2; ++i)
    for (k = 0; k < argc; k++)
      ;
  foo();
// CHECK: @{{.+}}foo
// CHECK-NOT: @k{{.+}}!llvm.access.group
// CHECK: i32 @{{.+}}bar{{.+}}!llvm.access.group
// CHECK-NOT: @k{{.+}}!llvm.access.group
// CHECK: sdiv i32
// CHECK: store i32 %{{.+}}, ptr @k,
#pragma omp simd linear(k : 2)
  for (k = 0; k < argc; k++)
    bar();
// CHECK: @{{.+}}foo
// CHECK-NOT: @k{{.+}}!llvm.access.group
// CHECK: i32 @{{.+}}bar{{.+}}!llvm.access.group
// CHECK-NOT: @k{{.+}}!llvm.access.group
// CHECK: sdiv i32
// CHECK: store i32 %{{.+}}, ptr @k,
  foo();
#pragma omp simd lastprivate(k) collapse(2)
  for (int i = 0; i < 2; ++i)
    for (k = 0; k < argc; k++)
     bar() ;
  foo();
// CHECK: @{{.+}}foo
// CHECK-NOT: @k{{.+}}!llvm.access.group
// CHECK: i32 @{{.+}}bar{{.+}}!llvm.access.group
// CHECK-NOT: @k{{.+}}!llvm.access.group
// CHECK: sdiv i32
// CHECK: store i32 %{{.+}}, ptr @k,
#pragma omp simd
  for (k = 0; k < argc; k++)
    bar();
  foo();
// CHECK: @{{.+}}foo
// CHECK-NOT: @k{{.+}}!llvm.access.group
// CHECK: i32 @{{.+}}bar{{.+}}!llvm.access.group
// CHECK-NOT: @k{{.+}}!llvm.access.group
// CHECK: sdiv i32
// CHECK: store i32 %{{.+}}, ptr @k,
#pragma omp simd collapse(2)
  for (int i = 0; i < 2; ++i)
    for (k = 0; k < argc; k++)
      bar();
// CHECK: @{{.+}}foo
  foo();
  return 0;
}

struct S {
  int k;
  S(int argc) {
  foo();
// CHECK: @{{.+}}foo
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK-NOT: getelementptr inbounds %struct.S, ptr %{{.+}}, i32 0, i32 0
// CHECK: call void @__kmpc_for_static_fini(
// CHECK-NOT: getelementptr inbounds %struct.S, ptr %{{.+}}, i32 0, i32 0
#pragma omp for private(k)
    for (k = 0; k < argc; k++)
      ;
  foo();
// CHECK: @{{.+}}foo
// CHECK: call void @__kmpc_for_static_init_8(
// CHECK-NOT: getelementptr inbounds %struct.S, ptr %{{.+}}, i32 0, i32 0
// CHECK: call void @__kmpc_for_static_fini(
#pragma omp for lastprivate(k) collapse(2)
    for (int i = 0; i < 2; ++i)
      for (k = 0; k < argc; k++)
        ;
  foo();
// CHECK: @{{.+}}foo
// CHECK: getelementptr inbounds %struct.S, ptr %{{.+}}, i32 0, i32 0
// CHECK: br i1
// CHECK-NOT: getelementptr inbounds %struct.S, ptr %{{.+}}, i32 0, i32 0
// CHECK: i32 @{{.+}}bar{{.+}}!llvm.access.group
// CHECK-NOT: getelementptr inbounds %struct.S, ptr %{{.+}}, i32 0, i32 0
// CHECK: add nsw i32 %{{.+}}, 1
// CHECK: br label {{.+}}, !llvm.loop
// CHECK: getelementptr inbounds %struct.S, ptr %{{.+}}, i32 0, i32 0
#pragma omp simd linear(k : 2)
    for (k = 0; k < argc; k++)
      bar();
  foo();
// CHECK: @{{.+}}foo
// CHECK: getelementptr inbounds %struct.S, ptr %{{.+}}, i32 0, i32 0
// CHECK: br i1
// CHECK-NOT: getelementptr inbounds %struct.S, ptr %{{.+}}, i32 0, i32 0
// CHECK: i32 @{{.+}}bar{{.+}}!llvm.access.group
// CHECK-NOT: getelementptr inbounds %struct.S, ptr %{{.+}}, i32 0, i32 0
// CHECK: add nsw i64 %{{.+}}, 1
// CHECK: br label {{.+}}, !llvm.loop
// CHECK: getelementptr inbounds %struct.S, ptr %{{.+}}, i32 0, i32 0
#pragma omp simd lastprivate(k) collapse(2)
    for (int i = 0; i < 2; ++i)
      for (k = 0; k < argc; k++)
        bar();
  foo();
// CHECK: @{{.+}}foo
// CHECK-NOT: getelementptr inbounds %struct.S, ptr %{{.+}}, i32 0, i32 0
// CHECK: br i1
// CHECK-NOT: getelementptr inbounds %struct.S, ptr %{{.+}}, i32 0, i32 0
// CHECK: i32 @{{.+}}bar{{.+}}!llvm.access.group
// CHECK-NOT: getelementptr inbounds %struct.S, ptr %{{.+}}, i32 0, i32 0
// CHECK: add nsw i32 %{{.+}}, 1
// CHECK: br label {{.+}}, !llvm.loop
// CHECK: getelementptr inbounds %struct.S, ptr %{{.+}}, i32 0, i32 0
#pragma omp simd
    for (k = 0; k < argc; k++)
      bar();
  foo();
// CHECK: @{{.+}}foo
// CHECK-NOT: getelementptr inbounds %struct.S, ptr %{{.+}}, i32 0, i32 0
// CHECK: br i1
// CHECK-NOT: getelementptr inbounds %struct.S, ptr %{{.+}}, i32 0, i32 0
// CHECK: i32 @{{.+}}bar{{.+}}!llvm.access.group
// CHECK-NOT: getelementptr inbounds %struct.S, ptr %{{.+}}, i32 0, i32 0
// CHECK: add nsw i64 %{{.+}}, 1
// CHECK: br label {{.+}}, !llvm.loop
// CHECK: getelementptr inbounds %struct.S, ptr %{{.+}}, i32 0, i32 0
#pragma omp simd collapse(2)
    for (int i = 0; i < 2; ++i)
      for (k = 0; k < argc; k++)
        bar();
// CHECK: @{{.+}}foo
  foo();
  }
} s(N);

#endif // HEADER
