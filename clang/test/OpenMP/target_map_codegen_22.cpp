// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -std=c++11 -DCK23 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK23 --check-prefix CK23-64
// RUN: %clang_cc1 -std=c++11 -DCK23 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK23 --check-prefix CK23-64
// RUN: %clang_cc1 -std=c++11 -DCK23 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK23 --check-prefix CK23-32
// RUN: %clang_cc1 -std=c++11 -DCK23 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK23 --check-prefix CK23-32

// RUN: %clang_cc1 -std=c++11 -DCK23 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK23 --check-prefix CK23-64
// RUN: %clang_cc1 -std=c++11 -DCK23 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK23 --check-prefix CK23-64
// RUN: %clang_cc1 -std=c++11 -DCK23 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK23 --check-prefix CK23-32
// RUN: %clang_cc1 -std=c++11 -DCK23 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK23 --check-prefix CK23-32

// RUN: %clang_cc1 -std=c++11 -DCK23 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK23 --check-prefix CK23-64
// RUN: %clang_cc1 -std=c++11 -DCK23 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK23 --check-prefix CK23-64
// RUN: %clang_cc1 -std=c++11 -DCK23 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK23 --check-prefix CK23-32
// RUN: %clang_cc1 -std=c++11 -DCK23 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK23 --check-prefix CK23-32

// RUN: %clang_cc1 -std=c++11 -DCK23 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY22 %s
// RUN: %clang_cc1 -std=c++11 -DCK23 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY22 %s
// RUN: %clang_cc1 -std=c++11 -DCK23 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY22 %s
// RUN: %clang_cc1 -std=c++11 -DCK23 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY22 %s
// SIMD-ONLY22-NOT: {{__kmpc|__tgt}}
#ifdef CK23

// CK23-LABEL: @.__omp_offloading_{{.*}}explicit_maps_inside_captured{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK23: [[SIZE00:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK23: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK23-LABEL: @.__omp_offloading_{{.*}}explicit_maps_inside_captured{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK23: [[SIZE01:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK23: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK23-LABEL: @.__omp_offloading_{{.*}}explicit_maps_inside_captured{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK23: [[SIZE02:@.+]] = private {{.*}}constant [1 x i64] [i64 400]
// CK23: [[MTYPE02:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK23-LABEL: @.__omp_offloading_{{.*}}explicit_maps_inside_captured{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK23: [[SIZE03:@.+]] = private {{.*}}constant [1 x i64] [i64 {{8|4}}]
// CK23: [[MTYPE03:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK23-LABEL: @.__omp_offloading_{{.*}}explicit_maps_inside_captured{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK23: [[SIZE04:@.+]] = private {{.*}}constant [1 x i64] [i64 16]
// CK23: [[MTYPE04:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK23-LABEL: @.__omp_offloading_{{.*}}explicit_maps_inside_captured{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK23: [[SIZE05:@.+]] = private {{.*}}constant [1 x i64] [i64 16]
// CK23: [[MTYPE05:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK23-LABEL: explicit_maps_inside_captured{{.*}}(
int explicit_maps_inside_captured(int a){
  float b;
  float c[100];
  float *d;

  // CK23: call void @{{.*}}explicit_maps_inside_captured{{.*}}([[SA:%.+]]* {{.*}})
  // CK23: define {{.*}}explicit_maps_inside_captured{{.*}}
  [&](void){
    // Region 00
    // CK23-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
    // CK23-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK23-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK23-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
    // CK23-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
    // CK23-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
    // CK23-DAG: store i32* [[VAR00:%.+]], i32** [[CP0]]
    // CK23-DAG: [[VAR0]] = load i32*, i32** [[CAP0:%[^,]+]]
    // CK23-DAG: [[CAP0]] = getelementptr inbounds [[SA]], [[SA]]
    // CK23-DAG: [[VAR00]] = load i32*, i32** [[CAP00:%[^,]+]]
    // CK23-DAG: [[CAP00]] = getelementptr inbounds [[SA]], [[SA]]

    // CK23: call void [[CALL00:@.+]](i32* {{[^,]+}})
    #pragma omp target map(a)
      { a+=1; }
    // Region 01
    // CK23-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}}, i8** null)
    // CK23-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK23-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK23-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to float**
    // CK23-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to float**
    // CK23-DAG: store float* [[VAR0:%.+]], float** [[CBP0]]
    // CK23-DAG: store float* [[VAR00:%.+]], float** [[CP0]]
    // CK23-DAG: [[VAR0]] = load float*, float** [[CAP0:%[^,]+]]
    // CK23-DAG: [[CAP0]] = getelementptr inbounds [[SA]], [[SA]]
    // CK23-DAG: [[VAR00]] = load float*, float** [[CAP00:%[^,]+]]
    // CK23-DAG: [[CAP00]] = getelementptr inbounds [[SA]], [[SA]]

    // CK23: call void [[CALL01:@.+]](float* {{[^,]+}})
    #pragma omp target map(b)
      { b+=1; }
    // Region 02
    // CK23-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE02]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE02]]{{.+}}, i8** null)
    // CK23-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK23-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK23-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x float]**
    // CK23-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [100 x float]**
    // CK23-DAG: store [100 x float]* [[VAR0:%.+]], [100 x float]** [[CBP0]]
    // CK23-DAG: store [100 x float]* [[VAR00:%.+]], [100 x float]** [[CP0]]
    // CK23-DAG: [[VAR0]] = load [100 x float]*, [100 x float]** [[CAP0:%[^,]+]]
    // CK23-DAG: [[CAP0]] = getelementptr inbounds [[SA]], [[SA]]
    // CK23-DAG: [[VAR00]] = load [100 x float]*, [100 x float]** [[CAP00:%[^,]+]]
    // CK23-DAG: [[CAP00]] = getelementptr inbounds [[SA]], [[SA]]

    // CK23: call void [[CALL02:@.+]]([100 x float]* {{[^,]+}})
    #pragma omp target map(c)
      { c[3]+=1; }

    // Region 03
    // CK23-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE03]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE03]]{{.+}}, i8** null)
    // CK23-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK23-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK23-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to float***
    // CK23-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to float***
    // CK23-DAG: store float** [[VAR0:%.+]], float*** [[CBP0]]
    // CK23-DAG: store float** [[VAR00:%.+]], float*** [[CP0]]
    // CK23-DAG: [[VAR0]] = load float**, float*** [[CAP0:%[^,]+]]
    // CK23-DAG: [[CAP0]] = getelementptr inbounds [[SA]], [[SA]]
    // CK23-DAG: [[VAR00]] = load float**, float*** [[CAP00:%[^,]+]]
    // CK23-DAG: [[CAP00]] = getelementptr inbounds [[SA]], [[SA]]

    // CK23: call void [[CALL03:@.+]](float** {{[^,]+}})
    #pragma omp target map(d)
      { d[3]+=1; }
    // Region 04
    // CK23-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE04]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE04]]{{.+}}, i8** null)
    // CK23-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK23-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK23-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x float]**
    // CK23-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to float**
    // CK23-DAG: store [100 x float]* [[VAR0:%.+]], [100 x float]** [[CBP0]]
    // CK23-DAG: store float* [[SEC0:%.+]], float** [[CP0]]
    // CK23-DAG: [[SEC0]] = getelementptr {{.*}}[100 x float]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 2
    // CK23-DAG: [[VAR0]] = load [100 x float]*, [100 x float]** [[CAP0:%[^,]+]]
    // CK23-DAG: [[CAP0]] = getelementptr inbounds [[SA]], [[SA]]
    // CK23-DAG: [[VAR00]] = load [100 x float]*, [100 x float]** [[CAP00:%[^,]+]]
    // CK23-DAG: [[CAP00]] = getelementptr inbounds [[SA]], [[SA]]

    // CK23: call void [[CALL04:@.+]]([100 x float]* {{[^,]+}})
    #pragma omp target map(c[2:4])
      { c[3]+=1; }

    // Region 05
    // CK23-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE05]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE05]]{{.+}}, i8** null)
    // CK23-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK23-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK23-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to float**
    // CK23-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to float**
    // CK23-DAG: store float* [[RVAR0:%.+]], float** [[CBP0]]
    // CK23-DAG: store float* [[SEC0:%.+]], float** [[CP0]]
    // CK23-DAG: [[RVAR0]] = load float*, float** [[VAR0:%[^,]+]]
    // CK23-DAG: [[SEC0]] = getelementptr {{.*}}float* [[RVAR00:%.+]], i{{.+}} 2
    // CK23-DAG: [[RVAR00]] = load float*, float** [[VAR00:%[^,]+]]
    // CK23-DAG: [[VAR0]] = load float**, float*** [[CAP0:%[^,]+]]
    // CK23-DAG: [[CAP0]] = getelementptr inbounds [[SA]], [[SA]]
    // CK23-DAG: [[VAR00]] = load float**, float*** [[CAP00:%[^,]+]]
    // CK23-DAG: [[CAP00]] = getelementptr inbounds [[SA]], [[SA]]

    // CK23: call void [[CALL05:@.+]](float* {{[^,]+}})
    #pragma omp target map(d[2:4])
      { d[3]+=1; }
  }();
  return b;
}

// CK23: define {{.+}}[[CALL00]]
// CK23: define {{.+}}[[CALL01]]
// CK23: define {{.+}}[[CALL02]]
// CK23: define {{.+}}[[CALL03]]
// CK23: define {{.+}}[[CALL04]]
// CK23: define {{.+}}[[CALL05]]
#endif // CK23
#endif
