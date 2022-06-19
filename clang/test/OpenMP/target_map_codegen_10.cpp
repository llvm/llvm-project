// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -no-opaque-pointers -DCK11 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK11
// RUN: %clang_cc1 -no-opaque-pointers -DCK11 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK11
// RUN: %clang_cc1 -no-opaque-pointers -DCK11 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK11
// RUN: %clang_cc1 -no-opaque-pointers -DCK11 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK11

// RUN: %clang_cc1 -no-opaque-pointers -DCK11 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY10 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK11 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY10 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK11 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY10 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK11 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY10 %s
// SIMD-ONLY10-NOT: {{__kmpc|__tgt}}
#ifdef CK11

// CK11-LABEL: @.__omp_offloading_{{.*}}implicit_maps_double_complex{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK11-DAG: [[SIZES:@.+]] = {{.+}}constant [2 x i64] [i64 16, i64 {{8|4}}]
// Map types: OMP_MAP_TO  | OMP_MAP_FROM | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 547
// CK11-DAG: [[TYPES:@.+]] = {{.+}}constant [2 x i64] [i64 547, i64 547]

// CK11-LABEL: implicit_maps_double_complex{{.*}}(
void implicit_maps_double_complex (int a, int *b){
  double _Complex dc = (double)a;

  // CK11-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{.+}}, i8* {{.+}}, i32 2, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null, i8** null)
  // CK11-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK11-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK11-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK11-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK11-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to { double, double }**
  // CK11-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to { double, double }**
  // CK11-DAG: store { double, double }* [[PTR:%[^,]+]], { double, double }** [[CBP1]]
  // CK11-DAG: store { double, double }* [[PTR]], { double, double }** [[CP1]]

  // CK11: call void [[KERNEL:@.+]]({ double, double }* [[PTR]], i32** %{{.+}})
  #pragma omp target defaultmap(tofrom:scalar)
  {
   dc *= dc; *b = 1;
  }
}

// CK11: define internal void [[KERNEL]]({ double, double }* {{.*}}[[ARG:%.+]], i32** {{.*}})
// CK11: [[ADDR:%.+]] = alloca { double, double }*,
// CK11: store { double, double }* [[ARG]], { double, double }** [[ADDR]],
// CK11: [[REF:%.+]] = load { double, double }*, { double, double }** [[ADDR]],
// CK11: {{.+}} = getelementptr inbounds { double, double }, { double, double }* [[REF]], i32 0, i32 0
#endif // CK11
#endif
