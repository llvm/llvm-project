// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK20 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK20 --check-prefix CK20-64
// RUN: %clang_cc1 -DCK20 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20 --check-prefix CK20-64
// RUN: %clang_cc1 -DCK20 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20 --check-prefix CK20-32
// RUN: %clang_cc1 -DCK20 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20 --check-prefix CK20-32

// RUN: %clang_cc1 -DCK20 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK20 --check-prefix CK20-64
// RUN: %clang_cc1 -DCK20 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20 --check-prefix CK20-64
// RUN: %clang_cc1 -DCK20 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20 --check-prefix CK20-32
// RUN: %clang_cc1 -DCK20 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20 --check-prefix CK20-32

// RUN: %clang_cc1 -DCK20 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK20 --check-prefix CK20-64
// RUN: %clang_cc1 -DCK20 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20 --check-prefix CK20-64
// RUN: %clang_cc1 -DCK20 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20 --check-prefix CK20-32
// RUN: %clang_cc1 -DCK20 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20 --check-prefix CK20-32

// RUN: %clang_cc1 -DCK20 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY19 %s
// RUN: %clang_cc1 -DCK20 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY19 %s
// RUN: %clang_cc1 -DCK20 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY19 %s
// RUN: %clang_cc1 -DCK20 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY19 %s
// SIMD-ONLY19-NOT: {{__kmpc|__tgt}}
#ifdef CK20

// CK20-LABEL: @.__omp_offloading_{{.*}}explicit_maps_references_and_function_args{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK20: [[SIZE00:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK20: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i64] [i64 33]

// CK20-LABEL: @.__omp_offloading_{{.*}}explicit_maps_references_and_function_args{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK20: [[SIZE01:@.+]] = private {{.*}}constant [1 x i64] [i64 20]
// CK20: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i64] [i64 33]

// CK20-LABEL: @.__omp_offloading_{{.*}}explicit_maps_references_and_function_args{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK20: [[SIZE02:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK20: [[MTYPE02:@.+]] = private {{.*}}constant [1 x i64] [i64 34]

// CK20-LABEL: @.__omp_offloading_{{.*}}explicit_maps_references_and_function_args{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK20: [[SIZE03:@.+]] = private {{.*}}constant [1 x i64] [i64 12]
// CK20: [[MTYPE03:@.+]] = private {{.*}}constant [1 x i64] [i64 34]

// CK20-LABEL: explicit_maps_references_and_function_args{{.*}}(
void explicit_maps_references_and_function_args (int a, float b, int (&c)[10], float *d){

  int &aa = a;
  float &bb = b;
  int (&cc)[10] = c;
  float *&dd = d;

  // Region 00
  // CK20-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null, i8** null)
  // CK20-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK20-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK20-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK20-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK20-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK20-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK20-DAG: store i32* [[RVAR0:%.+]], i32** [[CBP0]]
  // CK20-DAG: store i32* [[RVAR00:%.+]], i32** [[CP0]]
  // CK20-DAG: [[RVAR0]] = load i32*, i32** [[VAR0:%[^,]+]]
  // CK20-DAG: [[RVAR00]] = load i32*, i32** [[VAR0]]

  // CK20: call void [[CALL00:@.+]](i32* {{[^,]+}})
  #pragma omp target map(to:aa)
  {
    aa += 1;
  }

  // Region 01
  // CK20-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}}, i8** null, i8** null)
  // CK20-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK20-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK20-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK20-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK20-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [10 x i32]**
  // CK20-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK20-DAG: store [10 x i32]* [[RVAR0:%.+]], [10 x i32]** [[CBP0]]
  // CK20-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK20-DAG: [[SEC0]] = getelementptr {{.*}}[10 x i32]* [[RVAR00:%.+]], i{{.+}} 0, i{{.+}} 0
  // CK20-DAG: [[RVAR0]] = load [10 x i32]*, [10 x i32]** [[VAR0:%[^,]+]]
  // CK20-DAG: [[RVAR00]] = load [10 x i32]*, [10 x i32]** [[VAR0]]

  // CK20: call void [[CALL01:@.+]]([10 x i32]* {{[^,]+}})
  #pragma omp target map(to:cc[:5])
  {
    cc[3] += 1;
  }

  // Region 02
  // CK20-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE02]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE02]]{{.+}}, i8** null, i8** null)
  // CK20-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK20-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK20-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK20-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK20-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to float**
  // CK20-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to float**
  // CK20-DAG: store float* [[VAR0:%.+]], float** [[CBP0]]
  // CK20-DAG: store float* [[VAR0]], float** [[CP0]]

  // CK20: call void [[CALL02:@.+]](float* {{[^,]+}})
  #pragma omp target map(from:b)
  {
    b += 1.0f;
  }

  // Region 03
  // CK20-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE03]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE03]]{{.+}}, i8** null, i8** null)
  // CK20-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK20-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK20-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK20-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK20-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to float**
  // CK20-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to float**
  // CK20-DAG: store float* [[RVAR0:%.+]], float** [[CBP0]]
  // CK20-DAG: store float* [[SEC0:%.+]], float** [[CP0]]
  // CK20-DAG: [[RVAR0]] = load float*, float** [[VAR0:%[^,]+]]
  // CK20-DAG: [[SEC0]] = getelementptr {{.*}}float* [[RVAR00:%.+]], i{{.+}} 2
  // CK20-DAG: [[RVAR00]] = load float*, float** [[VAR0]]

  // CK20: call void [[CALL03:@.+]](float* {{[^,]+}})
  #pragma omp target map(from:d[2:3])
  {
    d[2] += 1.0f;
  }
}

// CK20: define {{.+}}[[CALL00]]
// CK20: define {{.+}}[[CALL01]]
// CK20: define {{.+}}[[CALL02]]
// CK20: define {{.+}}[[CALL03]]

#endif // CK20
#endif
