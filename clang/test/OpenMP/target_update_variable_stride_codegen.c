// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK --check-prefix CK-64
// RUN: %clang_cc1 -DCK -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK --check-prefix CK-32

// RUN: %clang_cc1 -DCK -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
#ifdef CK

// Test that variable stride expressions in target update directives correctly
// set the OMP_MAP_NON_CONTIG flag (0x100000000000) in the generated IR.
// OMP_MAP_TO = 0x1, OMP_MAP_FROM = 0x2
// NON_CONTIG | TO = 0x100000000001, NON_CONTIG | FROM = 0x100000000002

extern int get_stride();

// CK-64-DAG: [[MTYPE_VAR_STRIDE_TO:@.+]] = {{.+}}constant [1 x i64] [i64 [[#0x100000000001]]]
// CK-32-DAG: [[MTYPE_VAR_STRIDE_TO:@.+]] = {{.+}}constant [1 x i64] [i64 [[#0x100000000001]]]
// CK-64-DAG: [[MTYPE_VAR_STRIDE_FROM:@.+]] = {{.+}}constant [1 x i64] [i64 [[#0x100000000002]]]
// CK-32-DAG: [[MTYPE_VAR_STRIDE_FROM:@.+]] = {{.+}}constant [1 x i64] [i64 [[#0x100000000002]]]
// CK-64-DAG: [[MTYPE_CONST_STRIDE_ONE:@.+]] = {{.+}}constant [1 x i64] [i64 [[#0x1]]]
// CK-32-DAG: [[MTYPE_CONST_STRIDE_ONE:@.+]] = {{.+}}constant [1 x i64] [i64 [[#0x1]]]

void test_variable_stride_to() {
  int stride = get_stride();
  int data[10];
  // CK-DAG: call void @__tgt_target_data_update_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr {{%.+}}, ptr {{%.+}}, ptr @.offload_sizes, ptr [[MTYPE_VAR_STRIDE_TO]], ptr null, ptr null)
  #pragma omp target update to(data[0:5:stride])
}

void test_variable_stride_from() {
  int stride = get_stride();
  int data[10];
  // CK-DAG: call void @__tgt_target_data_update_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr {{%.+}}, ptr {{%.+}}, ptr @.offload_sizes{{.*}}, ptr [[MTYPE_VAR_STRIDE_FROM]], ptr null, ptr null)
  #pragma omp target update from(data[0:5:stride])
}

void test_constant_stride_one() {
  int data[10];
  // CK-DAG: call void @__tgt_target_data_update_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr {{%.+}}, ptr {{%.+}}, ptr @.offload_sizes{{.*}}, ptr [[MTYPE_CONST_STRIDE_ONE]], ptr null, ptr null)
  #pragma omp target update to(data[0:5:1])
}

#endif // CK
#endif // HEADER
