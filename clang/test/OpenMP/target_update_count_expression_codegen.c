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

// Test that OMPIRBuilder.cpp correctly stores dimension count (not byte size)
// in offload_sizes for NON_CONTIG entries with count expressions.
// 
// This test verifies the IR change: offload_sizes contains [i64 2] for
// NON_CONTIG entries (dimension count) and [i64 20] for non-NON_CONTIG entries
// (byte size: 5 elements * 4 bytes).

// CK-64-DAG: [[SIZE_NON_CONTIG:@.+]] = {{.+}}constant [1 x i64] [i64 2]
// CK-32-DAG: [[SIZE_NON_CONTIG:@.+]] = {{.+}}constant [1 x i64] [i64 2]
// For non-NON_CONTIG entries, offload_sizes contains byte size (5 elements * 4 bytes = 20)
// CK-64-DAG: [[SIZE_CONTIG:@.+]] = {{.+}}constant [1 x i64] [i64 20]
// CK-32-DAG: [[SIZE_CONTIG:@.+]] = {{.+}}constant [1 x i64] [i64 20]

void test_non_contig_dimension_count() {
  const int len = 10;
  int data[10];
  // CK-DAG: call void @__tgt_target_data_update_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr {{%.+}}, ptr {{%.+}}, ptr [[SIZE_NON_CONTIG]], ptr {{.+}}, ptr null, ptr null)
  #pragma omp target update to(data[0:len/2:2])
}

void test_contig_byte_size() {
  int data[10];
  // CK-DAG: call void @__tgt_target_data_update_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr {{%.+}}, ptr {{%.+}}, ptr [[SIZE_CONTIG]], ptr {{.+}}, ptr null, ptr null)
  #pragma omp target update to(data[0:5:1])
}

#endif // CK
#endif // HEADER
