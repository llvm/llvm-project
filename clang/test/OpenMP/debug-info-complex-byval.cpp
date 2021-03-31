// RUN: %clang_cc1 -fopenmp -x c++ %s -verify -debug-info-kind=limited -triple x86_64-unknown-unknown -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -fopenmp-simd -x c++ %s -verify -debug-info-kind=limited -triple x86_64-unknown-unknown -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics

void a() {
  float _Complex b;
  // CHECK: call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* {{.*}}, i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i64)* [[OUTLINED:@.+]] to void (i32*, i32*, ...)*), i64 %{{.*}})
#pragma omp parallel firstprivate(b)
  ;
}

// CHECK: define internal void [[OUTLINED_DEBUG:@.+]](i32* {{.*}}, i32* {{.*}}, <2 x float> {{.*}})

// CHECK: define internal void [[OUTLINED]](i32* {{.*}}, i32* {{.*}}, i64 [[B_VAL:%.+]])
// CHECK: [[B_ADDR:%.+]] = alloca i64,
// CHECK: store i64 [[B_VAL]], i64* [[B_ADDR]],
// CHECK: [[CONV:%.+]] = bitcast i64* [[B_ADDR]] to { float, float }*,
// CHECK: [[BC:%.+]] = bitcast { float, float }* [[CONV]] to <2 x float>*,
// CHECK: [[B_VAL:%.+]] = load <2 x float>, <2 x float>* [[BC]],
// CHECK: call void [[OUTLINED_DEBUG]](i32* %{{.+}}, i32* %{{.+}}, <2 x float> [[B_VAL]])
