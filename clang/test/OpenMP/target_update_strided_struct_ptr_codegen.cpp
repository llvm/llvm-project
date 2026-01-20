// Test codegen for strided target update with struct containing pointer-to-array member
// RUN: %clang_cc1 -DCK27 -verify -Wno-vla -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK27
// RUN: %clang_cc1 -DCK27 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck %s --check-prefix CK27

// RUN: %clang_cc1 -DCK27 -verify -Wno-vla -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK27 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics

#ifdef CK27
#ifndef CK27_INCLUDED
#define CK27_INCLUDED

// Verify that non-contiguous map type flag is set (bit 48)
// 17592186044418 = 0x1000000000002 (OMP_MAP_NON_CONTIG | OMP_MAP_FROM)
// 17592186044417 = 0x1000000000001 (OMP_MAP_NON_CONTIG | OMP_MAP_TO)
// CK27-DAG: @.offload_maptypes{{.*}} = private unnamed_addr constant [1 x i64] [i64 17592186044418]
// CK27-DAG: @.offload_maptypes{{.*}} = private unnamed_addr constant [1 x i64] [i64 17592186044417]

struct T {
  double *data;
  int len;
};

// CK27-LABEL: define {{.*}}void @{{.*}}test_strided_update_from{{.*}}(
void test_strided_update_from(int arg) {
  T s;
  s.len = 16;
  s.data = new double[16];

  for (int i = 0; i < 16; i++) {
    s.data[i] = i;
  }

  // Verify the stride descriptor is created with correct values:
  // - offset = 0
  // - count = 4 (number of elements to update)
  // - stride = 16 (2 * sizeof(double) = 2 * 8 = 16 bytes)
  // CK27-DAG: store i64 0, ptr %{{.+}}, align 8
  // CK27-DAG: store i64 4, ptr %{{.+}}, align 8
  // CK27-DAG: store i64 16, ptr %{{.+}}, align 8
  
  // Verify __tgt_target_data_update_mapper is called
  // CK27: call void @__tgt_target_data_update_mapper(ptr @{{.+}}, i64 -1, i32 {{1|signext 1}}, ptr %{{.+}}, ptr %{{.+}}, ptr @{{.+}}, ptr @.offload_maptypes{{.*}}, ptr null, ptr null)

  #pragma omp target update from(s.data[0:4:2])

  delete[] s.data;
}

// CK27-LABEL: define {{.*}}void @{{.*}}test_strided_update_to{{.*}}(
void test_strided_update_to(int arg) {
  T s;
  s.len = 16;
  s.data = new double[16];

  for (int i = 0; i < 16; i++) {
    s.data[i] = i;
  }

  // Verify __tgt_target_data_update_mapper is called with TO map type
  // CK27: call void @__tgt_target_data_update_mapper(ptr @{{.+}}, i64 -1, i32 {{1|signext 1}}, ptr %{{.+}}, ptr %{{.+}}, ptr @{{.+}}, ptr @.offload_maptypes{{.*}}, ptr null, ptr null)

  #pragma omp target update to(s.data[0:4:2])

  delete[] s.data;
}

#endif // CK27_INCLUDED
#endif // CK27
