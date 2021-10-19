// RUN: %clang_cc1 -verify -fopenmp -x c -emit-llvm %s -triple %itanium_abi_triple -o - -femit-all-decls -disable-llvm-passes | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c -triple %itanium_abi_triple -emit-pch -o %t %s -femit-all-decls -disable-llvm-passes
// RUN: %clang_cc1 -fopenmp -x c -triple %itanium_abi_triple -include-pch %t -verify %s -emit-llvm -o - -femit-all-decls -disable-llvm-passes | FileCheck --check-prefix=CHECK-LOAD %s

// RUN: %clang_cc1 -fopenmp -x c -triple %itanium_abi_triple -emit-pch -o %t %s -femit-all-decls -disable-llvm-passes -fopenmp-version=45
// RUN: %clang_cc1 -fopenmp -x c -triple %itanium_abi_triple -include-pch %t -verify %s -emit-llvm -o - -femit-all-decls -disable-llvm-passes -fopenmp-version=45 | FileCheck --check-prefixes=CHECK-LOAD,OMP45-LOAD %s

// RUN: %clang_cc1 -verify -fopenmp-simd -x c -emit-llvm %s -triple %itanium_abi_triple -o - -femit-all-decls -disable-llvm-passes | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c -triple %itanium_abi_triple -emit-pch -o %t %s -femit-all-decls -disable-llvm-passes
// RUN: %clang_cc1 -fopenmp-simd -x c -triple %itanium_abi_triple -include-pch %t -verify %s -emit-llvm -o - -femit-all-decls -disable-llvm-passes | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// CHECK: [[SSS_INT:.+]] = type { i32 }
// CHECK-LOAD: [[SSS_INT:.+]] = type { i32 }

// CHECK-DAG: [[SSS_INIT:@.+]] = private constant %struct.SSS zeroinitializer
// CHECK-DAG: [[INT_INIT:@.+]] = private constant i32 0

#pragma omp declare reduction(+ : int, char : omp_out *= omp_in)
// CHECK: define internal {{.*}}void @{{[^(]+}}(i32* noalias %0, i32* noalias %1)
// CHECK: [[MUL:%.+]] = mul nsw i32
// CHECK-NEXT: store i32 [[MUL]], i32*
// CHECK-NEXT: ret void
// CHECK-NEXT: }
// CHECK-LOAD: define internal {{.*}}void @{{[^(]+}}(i32* noalias %0, i32* noalias %1)
// CHECK-LOAD: [[MUL:%.+]] = mul nsw i32
// CHECK-LOAD-NEXT: store i32 [[MUL]], i32*
// CHECK-LOAD-NEXT: ret void
// CHECK-LOAD-NEXT: }

// CHECK: define internal {{.*}}void @{{[^(]+}}(i8* noalias %0, i8* noalias %1)
// CHECK: sext i8
// CHECK: sext i8
// CHECK: [[MUL:%.+]] = mul nsw i32
// CHECK-NEXT: [[TRUNC:%.+]] = trunc i32 [[MUL]] to i8
// CHECK-NEXT: store i8 [[TRUNC]], i8*
// CHECK-NEXT: ret void
// CHECK-NEXT: }
// CHECK-LOAD: define internal {{.*}}void @{{[^(]+}}(i8* noalias %0, i8* noalias %1)
// CHECK-LOAD: sext i8
// CHECK-LOAD: sext i8
// CHECK-LOAD: [[MUL:%.+]] = mul nsw i32
// CHECK-LOAD-NEXT: [[TRUNC:%.+]] = trunc i32 [[MUL]] to i8
// CHECK-LOAD-NEXT: store i8 [[TRUNC]], i8*
// CHECK-LOAD-NEXT: ret void
// CHECK-LOAD-NEXT: }

#pragma omp declare reduction(fun : float : omp_out += omp_in) initializer(omp_priv = 15 + omp_orig)
// CHECK: define internal {{.*}}void @{{[^(]+}}(float* noalias %0, float* noalias %1)
// CHECK: [[ADD:%.+]] = fadd float
// CHECK-NEXT: store float [[ADD]], float*
// CHECK-NEXT: ret void
// CHECK-NEXT: }
// CHECK: define internal {{.*}}void @{{[^(]+}}(float* noalias %0, float* noalias %1)
// CHECK: [[ADD:%.+]] = fadd float 1.5
// CHECK-NEXT: store float [[ADD]], float*
// CHECK-NEXT: ret void
// CHECK-NEXT: }
// CHECK-LOAD: define internal {{.*}}void @{{[^(]+}}(float* noalias %0, float* noalias %1)
// CHECK-LOAD: [[ADD:%.+]] = fadd float
// CHECK-LOAD-NEXT: store float [[ADD]], float*
// CHECK-LOAD-NEXT: ret void
// CHECK-LOAD-NEXT: }
// CHECK-LOAD: define internal {{.*}}void @{{[^(]+}}(float* noalias %0, float* noalias %1)
// CHECK-LOAD: [[ADD:%.+]] = fadd float 1.5
// CHECK-LOAD-NEXT: store float [[ADD]], float*
// CHECK-LOAD-NEXT: ret void
// CHECK-LOAD-NEXT: }

struct SSS {
  int field;
#pragma omp declare reduction(+ : int, char : omp_out *= omp_in)
  // CHECK: define internal {{.*}}void @{{[^(]+}}(i32* noalias %0, i32* noalias %1)
  // CHECK: [[MUL:%.+]] = mul nsw i32
  // CHECK-NEXT: store i32 [[MUL]], i32*
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }

  // CHECK: define internal {{.*}}void @{{[^(]+}}(i8* noalias %0, i8* noalias %1)
  // CHECK: sext i8
  // CHECK: sext i8
  // CHECK: [[MUL:%.+]] = mul nsw i32
  // CHECK-NEXT: [[TRUNC:%.+]] = trunc i32 [[MUL]] to i8
  // CHECK-NEXT: store i8 [[TRUNC]], i8*
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
};

void init(struct SSS *priv, struct SSS orig);

#pragma omp declare reduction(fun : struct SSS : omp_out = omp_in) initializer(init(&omp_priv, omp_orig))
// CHECK: define internal {{.*}}void @{{[^(]+}}([[SSS_INT]]* noalias %0, [[SSS_INT]]* noalias %1)
// CHECK: call void @llvm.memcpy
// CHECK-NEXT: ret void
// CHECK-NEXT: }
// CHECK: define internal {{.*}}void @{{[^(]+}}([[SSS_INT]]* noalias %0, [[SSS_INT]]* noalias %1)
// CHECK: call void @init(
// CHECK-NEXT: ret void
// CHECK-NEXT: }
// CHECK-LOAD: define internal {{.*}}void @{{[^(]+}}([[SSS_INT]]* noalias %0, [[SSS_INT]]* noalias %1)
// CHECK-LOAD: call void @llvm.memcpy
// CHECK-LOAD-NEXT: ret void
// CHECK-LOAD-NEXT: }
// CHECK-LOAD: define internal {{.*}}void @{{[^(]+}}([[SSS_INT]]* noalias %0, [[SSS_INT]]* noalias %1)
// CHECK-LOAD: call void @init(
// CHECK-LOAD-NEXT: ret void
// CHECK-LOAD-NEXT: }

// CHECK-LABEL: @main
// CHECK-LOAD-LABEL: @main
int main() {
#pragma omp declare reduction(fun : struct SSS : omp_out = omp_in) initializer(init(&omp_priv, omp_orig))
  // CHECK: define internal {{.*}}void @{{[^(]+}}([[SSS_INT]]* noalias %0, [[SSS_INT]]* noalias %1)
  // CHECK: call void @llvm.memcpy
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  // CHECK: define internal {{.*}}void @{{[^(]+}}([[SSS_INT]]* noalias %0, [[SSS_INT]]* noalias %1)
  // CHECK: call void @init(
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  // CHECK-LOAD: define internal {{.*}}void @{{[^(]+}}([[SSS_INT]]* noalias %0, [[SSS_INT]]* noalias %1)
  // CHECK-LOAD: call void @llvm.memcpy
  // CHECK-LOAD-NEXT: ret void
  // CHECK-LOAD-NEXT: }
  // CHECK-LOAD: define internal {{.*}}void @{{[^(]+}}([[SSS_INT]]* noalias %0, [[SSS_INT]]* noalias %1)
  // CHECK-LOAD: call void @init(
  // CHECK-LOAD-NEXT: ret void
  // CHECK-LOAD-NEXT: }
  {
#pragma omp declare reduction(fun : struct SSS : omp_out = omp_in) initializer(init(&omp_priv, omp_orig))
    // CHECK: define internal {{.*}}void @{{[^(]+}}([[SSS_INT]]* noalias %0, [[SSS_INT]]* noalias %1)
    // CHECK: call void @llvm.memcpy
    // CHECK-NEXT: ret void
    // CHECK-NEXT: }
    // CHECK: define internal {{.*}}void @{{[^(]+}}([[SSS_INT]]* noalias %0, [[SSS_INT]]* noalias %1)
    // CHECK: call void @init(
    // CHECK-NEXT: ret void
    // CHECK-NEXT: }
    // CHECK-LOAD: define internal {{.*}}void @{{[^(]+}}([[SSS_INT]]* noalias %0, [[SSS_INT]]* noalias %1)
    // CHECK-LOAD: call void @llvm.memcpy
    // CHECK-LOAD-NEXT: ret void
    // CHECK-LOAD-NEXT: }
    // CHECK-LOAD: define internal {{.*}}void @{{[^(]+}}([[SSS_INT]]* noalias %0, [[SSS_INT]]* noalias %1)
    // CHECK-LOAD: call void @init(
    // CHECK-LOAD-NEXT: ret void
    // CHECK-LOAD-NEXT: }
  }
  return 0;
}

// OMP45-LOAD: define internal {{.*}}void @{{[^(]+}}(i32* noalias %0, i32* noalias %1)
// OMP45-LOAD: [[MUL:%.+]] = mul nsw i32
// OMP45-LOAD-NEXT: store i32 [[MUL]], i32*
// OMP45-LOAD-NEXT: ret void
// OMP45-LOAD-NEXT: }

// OMP45-LOAD: define internal {{.*}}void @{{[^(]+}}(i8* noalias %0, i8* noalias %1)
// OMP45-LOAD: sext i8
// OMP45-LOAD: sext i8
// OMP45-LOAD: [[MUL:%.+]] = mul nsw i32
// OMP45-LOAD-NEXT: [[TRUNC:%.+]] = trunc i32 [[MUL]] to i8
// OMP45-LOAD-NEXT: store i8 [[TRUNC]], i8*
// OMP45-LOAD-NEXT: ret void
// OMP45-LOAD-NEXT: }

// CHECK-LABEL: bar
struct SSS ss;
int in;
void bar() {
  // CHECK: [[SS_PRIV:%.+]] = alloca %struct.SSS,
  // CHECK: [[IN_PRIV:%.+]] = alloca i32,
  // CHECK: [[BC:%.+]] = bitcast %struct.SSS* [[SS_PRIV]] to i8*
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i{{64|32}}(i8* {{.*}}[[BC]], i8* {{.*}}bitcast (%struct.SSS* [[SSS_INIT]] to i8*), i{{64|32}} 4, i1 false)
  // CHECK: [[IN_VAL:%.+]] = load i32, i32* [[INT_INIT]],
  // CHECK: store i32 [[IN_VAL]], i32* [[IN_PRIV]],
  // CHECK: call void @__kmpc_for_static_init_4(
#pragma omp declare reduction(+            \
                              : struct SSS \
                              : omp_out = omp_in)
#pragma omp declare reduction(+     \
                              : int \
                              : omp_out = omp_in)
#pragma omp for reduction(+ \
                          : ss, in)
  for (int i = 0; i < 10; ++i)
    ;
}
#endif
