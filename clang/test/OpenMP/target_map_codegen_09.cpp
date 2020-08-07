// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK10 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK10
// RUN: %clang_cc1 -DCK10 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK10
// RUN: %clang_cc1 -DCK10 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK10
// RUN: %clang_cc1 -DCK10 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK10

// RUN: %clang_cc1 -DCK10 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK10
// RUN: %clang_cc1 -DCK10 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK10
// RUN: %clang_cc1 -DCK10 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK10
// RUN: %clang_cc1 -DCK10 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK10

// RUN: %clang_cc1 -DCK10 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK10
// RUN: %clang_cc1 -DCK10 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK10
// RUN: %clang_cc1 -DCK10 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK10
// RUN: %clang_cc1 -DCK10 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK10

// RUN: %clang_cc1 -DCK10 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY9 %s
// RUN: %clang_cc1 -DCK10 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY9 %s
// RUN: %clang_cc1 -DCK10 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY9 %s
// RUN: %clang_cc1 -DCK10 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY9 %s
// SIMD-ONLY9-NOT: {{__kmpc|__tgt}}
#ifdef CK10

// CK10-LABEL: @.__omp_offloading_{{.*}}implicit_maps_pointer{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK10-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] zeroinitializer
// Map types: OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 544
// CK10-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 544]

// CK10-LABEL: implicit_maps_pointer{{.*}}(
void implicit_maps_pointer (){
  double *ddyn;

  // CK10-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK10-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK10-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK10-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK10-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK10-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to double**
  // CK10-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to double**
  // CK10-DAG: store double* [[PTR:%[^,]+]], double** [[CBP1]]
  // CK10-DAG: store double* [[PTR]], double** [[CP1]]

  // CK10: call void [[KERNEL:@.+]](double* [[PTR]])
  #pragma omp target
  {
    ddyn[0] += 1.0;
    ddyn[1] += 1.0;
  }
}

// CK10: define internal void [[KERNEL]](double* {{.*}}[[ARG:%.+]])
// CK10: [[ADDR:%.+]] = alloca double*,
// CK10: store double* [[ARG]], double** [[ADDR]],
// CK10: [[REF:%.+]] = load double*, double** [[ADDR]],
// CK10: {{.+}} = getelementptr inbounds double, double* [[REF]], i{{64|32}} 0

#endif // CK10
#endif
