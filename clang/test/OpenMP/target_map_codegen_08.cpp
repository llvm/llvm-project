// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -no-opaque-pointers -DCK9 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK9
// RUN: %clang_cc1 -no-opaque-pointers -DCK9 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK9
// RUN: %clang_cc1 -no-opaque-pointers -DCK9 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK9
// RUN: %clang_cc1 -no-opaque-pointers -DCK9 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK9

// RUN: %clang_cc1 -no-opaque-pointers -DCK9 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK9
// RUN: %clang_cc1 -no-opaque-pointers -DCK9 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK9
// RUN: %clang_cc1 -no-opaque-pointers -DCK9 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK9
// RUN: %clang_cc1 -no-opaque-pointers -DCK9 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK9

// RUN: %clang_cc1 -no-opaque-pointers -DCK9 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK9
// RUN: %clang_cc1 -no-opaque-pointers -DCK9 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK9
// RUN: %clang_cc1 -no-opaque-pointers -DCK9 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK9
// RUN: %clang_cc1 -no-opaque-pointers -DCK9 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK9

// RUN: %clang_cc1 -no-opaque-pointers -DCK9 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK9 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK9 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK9 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// SIMD-ONLY8-NOT: {{__kmpc|__tgt}}
#ifdef CK9

// CK9-LABEL: @.__omp_offloading_{{.*}}implicit_maps_array{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK9-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// Map types: OMP_MAP_TO + OMP_MAP_FROM + OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 547
// CK9-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 547]

// CK9-LABEL: implicit_maps_array{{.*}}(
void implicit_maps_array (int a){
  double darr[2] = {(double)a, (double)a};

  // CK9-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null, i8** null)
  // CK9-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK9-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK9-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK9-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK9-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [2 x double]**
  // CK9-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to [2 x double]**
  // CK9-DAG: store [2 x double]* [[DECL:%[^,]+]], [2 x double]** [[CBP1]]
  // CK9-DAG: store [2 x double]* [[DECL]], [2 x double]** [[CP1]]

  // CK9: call void [[KERNEL:@.+]]([2 x double]* [[DECL]])
  #pragma omp target
  {
    darr[0] += 1.0;
    darr[1] += 1.0;
  }
}

// CK9: define internal void [[KERNEL]]([2 x double]* {{.+}}[[ARG:%.+]])
// CK9: [[ADDR:%.+]] = alloca [2 x double]*,
// CK9: store [2 x double]* [[ARG]], [2 x double]** [[ADDR]],
// CK9: [[REF:%.+]] = load [2 x double]*, [2 x double]** [[ADDR]],
// CK9: {{.+}} = getelementptr inbounds [2 x double], [2 x double]* [[REF]], i{{64|32}} 0, i{{64|32}} 0
#endif // CK9
#endif
