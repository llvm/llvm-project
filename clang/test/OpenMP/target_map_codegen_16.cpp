// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -no-opaque-pointers -DCK17 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK17
// RUN: %clang_cc1 -no-opaque-pointers -DCK17 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK17
// RUN: %clang_cc1 -no-opaque-pointers -DCK17 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK17
// RUN: %clang_cc1 -no-opaque-pointers -DCK17 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK17

// RUN: %clang_cc1 -no-opaque-pointers -DCK17 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK17
// RUN: %clang_cc1 -no-opaque-pointers -DCK17 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK17
// RUN: %clang_cc1 -no-opaque-pointers -DCK17 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK17
// RUN: %clang_cc1 -no-opaque-pointers -DCK17 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK17

// RUN: %clang_cc1 -no-opaque-pointers -DCK17 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK17
// RUN: %clang_cc1 -no-opaque-pointers -DCK17 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK17
// RUN: %clang_cc1 -no-opaque-pointers -DCK17 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK17
// RUN: %clang_cc1 -no-opaque-pointers -DCK17 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK17

// RUN: %clang_cc1 -no-opaque-pointers -DCK17 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY16 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK17 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY16 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK17 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY16 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK17 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY16 %s
// SIMD-ONLY16-NOT: {{__kmpc|__tgt}}
#ifdef CK17

// CK17-DAG: [[ST:%.+]] = type { i32, double }
// CK17-LABEL: @.__omp_offloading_{{.*}}implicit_maps_struct{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK17-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 {{16|12}}]
// Map types: OMP_MAP_TO + OMP_MAP_FROM + OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 547
// CK17-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 547]

class SSS {
public:
  int a;
  double b;
};

// CK17-LABEL: implicit_maps_struct{{.*}}(
void implicit_maps_struct (int a){
  SSS s = {a, (double)a};

// CK17-DAG: call i32 @__tgt_target_kernel(%struct.ident_t* @{{.+}}, i64 -1, i32 -1, i32 0, i8* @.{{.+}}.region_id, %struct.__tgt_kernel_arguments* [[ARGS:%.+]])
// CK17-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK17-DAG: store i8** [[BPGEP:%.+]], i8*** [[BPARG]]
// CK17-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK17-DAG: store i8** [[PGEP:%.+]], i8*** [[PARG]]
// CK17-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
// CK17-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
// CK17-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
// CK17-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
// CK17-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[ST]]**
// CK17-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to [[ST]]**
// CK17-DAG: store [[ST]]* [[DECL:%.+]], [[ST]]** [[CBP1]]
// CK17-DAG: store [[ST]]* [[DECL]], [[ST]]** [[CP1]]

// CK17: call void [[KERNEL:@.+]]([[ST]]* [[DECL]])
#pragma omp target
  {
    s.a += 1;
    s.b += 1.0;
  }
}

// CK17: define internal void [[KERNEL]]([[ST]]* {{.+}}[[ARG:%.+]])
// CK17: [[ADDR:%.+]] = alloca [[ST]]*,
// CK17: store [[ST]]* [[ARG]], [[ST]]** [[ADDR]],
// CK17: [[REF:%.+]] = load [[ST]]*, [[ST]]** [[ADDR]],
// CK17: {{.+}} = getelementptr inbounds [[ST]], [[ST]]* [[REF]], i32 0, i32 0
#endif // CK17
#endif
