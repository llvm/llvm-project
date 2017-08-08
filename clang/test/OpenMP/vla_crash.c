// RUN: %clang_cc1 -verify -triple powerpc64le-unknown-linux-gnu -fopenmp -x c -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

int a;

// CHECK-LABEL: foo
void foo() {
  int(*b)[a];
  int *(**c)[a];
  // CHECK: [[B:%.+]] = alloca i32*,
  // CHECK: [[C:%.+]] = alloca i32***,
  // CHECK: @__kmpc_global_thread_num
  // CHECK: call void @__kmpc_serialized_parallel
  // CHECK: call void [[OUTLINED:@[^(]+]](i32* %{{[^,]+}}, i32* %{{[^,]+}}, i64 %{{[^,]+}}, i32** [[B]], i64 %{{[^,]+}}, i32**** [[C]])
  // CHECK: call void @__kmpc_end_serialized_parallel
  // CHECK: ret void
#pragma omp parallel if (0)
  b[0][0] = c[0][a][0][a];
}

// CHECK: define internal void [[OUTLINED]](i32* {{[^,]+}}, i32* {{[^,]+}}, i64 {{[^,]+}}, i32** {{[^,]+}}, i64 {{[^,]+}}, i32**** {{[^,]+}})

