// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -no-opaque-pointers -DCK1 -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK1 -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK1  --check-prefix CK1-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK1 -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK1  --check-prefix CK1-32
// RUN: %clang_cc1 -no-opaque-pointers -DCK1 -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK1  --check-prefix CK1-32

// RUN: %clang_cc1 -no-opaque-pointers -DCK1 -verify -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY6 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK1 -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY6 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK1 -verify -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY6 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK1 -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY6 %s
// SIMD-ONLY6-NOT: {{__kmpc|__tgt}}
#ifdef CK1

// CK1-LABEL: @.__omp_offloading_{{.*}}implicit_present_scalar{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK1-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 4]
// Map types: OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT | OMP_MAP_PRESENT = 4640
// CK1-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 4640]

// CK1-LABEL: implicit_present_scalar{{.*}}(
void implicit_present_scalar(int a) {
// CK1-DAG: call i32 @__tgt_target_kernel(%struct.ident_t* @{{.+}}, i64 -1, i32 -1, i32 0, i8* @.{{.+}}.region_id, %struct.__tgt_kernel_arguments* [[ARGS:%.+]])
// CK1-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK1-DAG: store i8** [[BPGEP:%.+]], i8*** [[BPARG]]
// CK1-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK1-DAG: store i8** [[PGEP:%.+]], i8*** [[PARG]]
// CK1-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
// CK1-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
// CK1-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
// CK1-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
// CK1-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32**
// CK1-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
// CK1-DAG: store i32* [[DECL:%[^,]+]], i32** [[CBP1]]
// CK1-DAG: store i32* [[DECL]], i32** [[CP1]]
#pragma omp target defaultmap(present \
                              : scalar)
  {
    a += 1.0;
  }
}


#endif
///==========================================================================///
// RUN: %clang_cc1 -no-opaque-pointers -DCK2 -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK2 -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK2  --check-prefix CK2-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK2 -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK2  --check-prefix CK2-32
// RUN: %clang_cc1 -no-opaque-pointers -DCK2 -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK2  --check-prefix CK2-32

// RUN: %clang_cc1 -no-opaque-pointers -DCK2 -verify -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY6 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK2 -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY6 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK2 -verify -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY6 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK2 -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY6 %s
// SIMD-ONLY6-NOT: {{__kmpc|__tgt}}
#ifdef CK2

// CK2-LABEL: @.__omp_offloading_{{.*}}implicit_present_aggregate{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK2-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// Map types: OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT | OMP_MAP_PRESENT = 4640
// CK2-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 4640]

// CK2-LABEL: implicit_present_aggregate{{.*}}(
void implicit_present_aggregate(int a) {
  double darr[2] = {(double)a, (double)a};

// CK2-DAG: call i32 @__tgt_target_kernel(%struct.ident_t* @{{.+}}, i64 -1, i32 -1, i32 0, i8* @.{{.+}}.region_id, %struct.__tgt_kernel_arguments* [[ARGS:%.+]])
// CK2-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK2-DAG: store i8** [[BPGEP:%.+]], i8*** [[BPARG]]
// CK2-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK2-DAG: store i8** [[PGEP:%.+]], i8*** [[PARG]]
// CK2-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
// CK2-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
// CK2-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
// CK2-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
// CK2-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [2 x double]**
// CK2-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to [2 x double]**
// CK2-DAG: store [2 x double]* [[DECL:%[^,]+]], [2 x double]** [[CBP1]]
// CK2-DAG: store [2 x double]* [[DECL]], [2 x double]** [[CP1]]

// CK2: call void [[KERNEL:@.+]]([2 x double]* [[DECL]])
#pragma omp target defaultmap(present \
                              : aggregate)
  {
    darr[0] += 1.0;
    darr[1] += 1.0;
  }
}


#endif
///==========================================================================///
// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK3
// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK3
// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK3
// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK3

// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -verify -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -verify -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// SIMD-ONLY8-NOT: {{__kmpc|__tgt}}
#ifdef CK3


// CK3-LABEL: @.__omp_offloading_{{.*}}explicit_present_pointer{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK3: [[SIZE:@.+]] = private {{.*}}constant [1 x i64] [i64 {{8|4}}]
// Map types: OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT | OMP_MAP_PRESENT = 4640
// CK3: [[MTYPE:@.+]] = private {{.*}}constant [1 x i64] [i64 4640]

// CK3-LABEL: explicit_present_pointer{{.*}}(
void explicit_present_pointer() {
  int *pa;

  // CK3-DAG: call i32 @__tgt_target_kernel(%struct.ident_t* @{{.+}}, i64 -1, i32 -1, i32 0, i8* @.{{.+}}.region_id, %struct.__tgt_kernel_arguments* [[ARGS:%.+]])
  // CK3-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
  // CK3-DAG: store i8** [[BPGEP:%.+]], i8*** [[BPARG]]
  // CK3-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
  // CK3-DAG: store i8** [[PGEP:%.+]], i8*** [[PARG]]
  // CK3-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK3-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK3-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK3-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK3-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32***
  // CK3-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32***
  // CK3-DAG: store i32** [[VAR0:%.+]], i32*** [[CBP0]]
  // CK3-DAG: store i32** [[VAR0]], i32*** [[CP0]]

  #pragma omp target defaultmap(present: pointer)
  {
    pa[50]++;
  }
}

#endif
#ifdef CK4

///==========================================================================///
// RUN: %clang_cc1 -no-opaque-pointers -DCK4 -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK4
// RUN: %clang_cc1 -no-opaque-pointers -DCK4 -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK4
// RUN: %clang_cc1 -no-opaque-pointers -DCK4 -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK4
// RUN: %clang_cc1 -no-opaque-pointers -DCK4 -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK4

// RUN: %clang_cc1 -no-opaque-pointers -DCK4 -verify -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY10 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK4 -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY10 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK4 -verify -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY10 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK4 -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY10 %s
// SIMD-ONLY10-NOT: {{__kmpc|__tgt}}

// CK4-LABEL: @.__omp_offloading_{{.*}}implicit_present_double_complex{{.*}}.region_id = weak constant i8 0

// CK4-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// Map types: OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT | OMP_MAP_PRESENT = 4640
// CK4-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 4640]

// CK4-LABEL: implicit_present_double_complex{{.*}}(
void implicit_present_double_complex (int a){
  double _Complex dc = (double)a;

// CK4-DAG: call i32 @__tgt_target_kernel(%struct.ident_t* @{{.+}}, i64 -1, i32 -1, i32 0, i8* @.{{.+}}.region_id, %struct.__tgt_kernel_arguments* [[ARGS:%.+]])
// CK4-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK4-DAG: store i8** [[BPGEP:%.+]], i8*** [[BPARG]]
// CK4-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK4-DAG: store i8** [[PGEP:%.+]], i8*** [[PARG]]
// CK4-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
// CK4-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
// CK4-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
// CK4-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
// CK4-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to { double, double }**
// CK4-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to { double, double }**
// CK4-DAG: store { double, double }* [[PTR:%[^,]+]], { double, double }** [[CBP1]]
// CK4-DAG: store { double, double }* [[PTR]], { double, double }** [[CP1]]

// CK4: call void [[KERNEL:@.+]]({ double, double }* [[PTR]])
#pragma omp target defaultmap(present \
                              : scalar)
  {
   dc *= dc;
  }
}

// CK4: define internal void [[KERNEL]]({ double, double }* {{.*}}[[ARG:%.+]])
// CK4: [[ADDR:%.+]] = alloca { double, double }*,
// CK4: store { double, double }* [[ARG]], { double, double }** [[ADDR]],
// CK4: [[REF:%.+]] = load { double, double }*, { double, double }** [[ADDR]],
// CK4: {{.+}} = getelementptr inbounds { double, double }, { double, double }* [[REF]], i32 0, i32 0
#endif
#endif
