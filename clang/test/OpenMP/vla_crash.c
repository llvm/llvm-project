// RUN: %clang_cc1 -verify -triple powerpc64le-unknown-linux-gnu -fopenmp -x c -emit-llvm %s -o - | FileCheck %s

// RUN: %clang_cc1 -verify -triple powerpc64le-unknown-linux-gnu -fopenmp-simd -x c -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

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

// CHECK-LABEL: bar
void bar(int n, int *a) {
  // CHECK: [[N:%.+]] = alloca i32,
  // CHECK: [[A:%.+]] = alloca i32*,
  // CHECK: [[P:%.+]] = alloca i32*,
  // CHECK: @__kmpc_global_thread_num
  // CHECK: [[BC:%.+]] = bitcast i32** [[A]] to i32*
  // CHECK: store i32* [[BC]], i32** [[P]],
  // CHECK: call void @__kmpc_serialized_parallel
  // CHECK: call void [[OUTLINED:@[^(]+]](i32* %{{[^,]+}}, i32* %{{[^,]+}}, i64 %{{[^,]+}}, i32** [[P]], i32** [[A]])
  // CHECK: call void @__kmpc_end_serialized_parallel
  // CHECK: ret void
  // expected-warning@+1 {{incompatible pointer types initializing 'int (*)[n]' with an expression of type 'int **'}}
  int(*p)[n] = &a;
#pragma omp parallel if(0)
  // expected-warning@+1 {{comparison of distinct pointer types ('int (*)[n]' and 'int **')}}
  if (p == &a) {
  }
}

// CHECK: define internal void [[OUTLINED]](i32* {{[^,]+}}, i32* {{[^,]+}}, i64 {{[^,]+}}, i32** {{[^,]+}}, i32** {{[^,]+}})
