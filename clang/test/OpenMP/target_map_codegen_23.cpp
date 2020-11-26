// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK24 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK24 --check-prefix CK24-64
// RUN: %clang_cc1 -DCK24 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-64
// RUN: %clang_cc1 -DCK24 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-32
// RUN: %clang_cc1 -DCK24 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-32

// RUN: %clang_cc1 -DCK24 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK24 --check-prefix CK24-64
// RUN: %clang_cc1 -DCK24 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-64
// RUN: %clang_cc1 -DCK24 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-32
// RUN: %clang_cc1 -DCK24 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-32

// RUN: %clang_cc1 -DCK24 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK24 --check-prefix CK24-64
// RUN: %clang_cc1 -DCK24 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-64
// RUN: %clang_cc1 -DCK24 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-32
// RUN: %clang_cc1 -DCK24 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-32

// RUN: %clang_cc1 -DCK24 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY23 %s
// RUN: %clang_cc1 -DCK24 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY23 %s
// RUN: %clang_cc1 -DCK24 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY23 %s
// RUN: %clang_cc1 -DCK24 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY23 %s
// SIMD-ONLY23-NOT: {{__kmpc|__tgt}}
#ifdef CK24

// CK24-DAG: [[SC:%.+]] = type { i32, [[SB:%.+]], [[SB:%.+]]*, [10 x i32] }
// CK24-DAG: [[SB]] = type { i32, [[SA:%.+]], [10 x [[SA:%.+]]], [10 x [[SA:%.+]]*], [[SA:%.+]]* }
// CK24-DAG: [[SA]] = type { i32, [[SA]]*, [10 x i32] }

struct SA{
  int a;
  struct SA *p;
  int b[10];
};
struct SB{
  int a;
  struct SA s;
  struct SA sa[10];
  struct SA *sp[10];
  struct SA *p;
};
struct SC{
  int a;
  struct SB s;
  struct SB *p;
  int b[10];
};

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[SIZE01:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK24: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[SIZE13:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK24: [[MTYPE13:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[SIZE14:@.+]] = private {{.*}}constant [1 x i64] [i64 {{48|56}}]
// CK24: [[MTYPE14:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[SIZE15:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK24: [[MTYPE15:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[MTYPE16:@.+]] = private {{.*}}constant [2 x i64] [i64 32, i64 281474976710659]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[MTYPE17:@.+]] = private {{.*}}constant [2 x i64] [i64 32, i64 281474976710675]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[SIZE18:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK24: [[MTYPE18:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[MTYPE19:@.+]] = private {{.*}}constant [3 x i64] [i64 32, i64 281474976710659, i64 281474976710675]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[MTYPE20:@.+]] = private {{.*}}constant [2 x i64] [i64 32, i64 281474976710675]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[MTYPE21:@.+]] = private {{.*}}constant [3 x i64] [i64 32, i64 281474976710659, i64 281474976710675]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[MTYPE22:@.+]] = private {{.*}}constant [2 x i64] [i64 32, i64 281474976710659]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[MTYPE23:@.+]] = private {{.*}}constant [3 x i64] [i64 32, i64 281474976710659, i64 281474976710675]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[MTYPE24:@.+]] = private {{.*}}constant [4 x i64] [i64 32, i64 281474976710672, i64 16, i64 19]

// CK24-LABEL: explicit_maps_struct_fields
int explicit_maps_struct_fields(int a){
  SC s;
  SC *p;

// Region 01
// CK24-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE01]]{{.+}}, {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}}, i8** null)
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SC]]**
// CK24-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
// CK24-DAG: store [[SC]]* [[VAR0:%.+]], [[SC]]** [[CBP0]]
// CK24-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SC]]* [[VAR0]], i{{.+}} 0, i{{.+}} 0

// CK24: call void [[CALL01:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(s.a)
  { s.a++; }

//
// Same thing but starting from a pointer.
//
// Region 13
// CK24-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE13]]{{.+}}, {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE13]]{{.+}}, i8** null)
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SC]]**
// CK24-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
// CK24-DAG: store [[SC]]* [[VAR0:%.+]], [[SC]]** [[CBP0]]
// CK24-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 0

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL13:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->a)
  { p->a++; }

// Region 14
// CK24-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE14]]{{.+}}, {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE14]]{{.+}}, i8** null)
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SC]]**
// CK24-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[SA]]**
// CK24-DAG: store [[SC]]* [[VAR0:%.+]], [[SC]]** [[CBP0]]
// CK24-DAG: store [[SA]]* [[SEC0:%.+]], [[SA]]** [[CP0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SB]]* [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL14:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->s.s)
  { p->a++; }

// Region 15
// CK24-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE15]]{{.+}}, {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE15]]{{.+}}, i8** null)
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SC]]**
// CK24-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
// CK24-DAG: store [[SC]]* [[VAR0:%.+]], [[SC]]** [[CBP0]]
// CK24-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SA]]* [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[[SB]]* [[SEC000:%[^,]+]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[SEC000]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL15:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->s.s.a)
  { p->a++; }

// Region 16
// CK24-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE16]]{{.+}}, i8** null)
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK24-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SC]]**
// CK24-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
// CK24-DAG: store [[SC]]* [[VAR0:%.+]], [[SC]]** [[CBP0]]
// CK24-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
// CK24-DAG: store i64 {{%.+}}, i64* [[S0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[10 x i32]* [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 3

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[SC]]**
// CK24-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
// CK24-DAG: store [[SC]]* [[VAR0]], [[SC]]** [[CBP1]]
// CK24-DAG: store i32* [[SEC0]], i32** [[CP1]]
// CK24-DAG: store i64 {{.+}}, i64* [[S1]]

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL16:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->b[:5])
  { p->a++; }

// Region 17
// CK24-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE17]]{{.+}}, i8** null)
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK24-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SC]]**
// CK24-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[SB]]***
// CK24-DAG: store [[SC]]* [[VAR0:%.+]], [[SC]]** [[CBP0]]
// CK24-DAG: store [[SB]]** [[SEC0:%.+]], [[SB]]*** [[CP0]]
// CK24-DAG: store i64 {{%.+}}, i64* [[S0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 2

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[SB]]***
// CK24-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to [[SB]]**
// CK24-DAG: store [[SB]]** [[SEC0]], [[SB]]*** [[CBP1]]
// CK24-DAG: store [[SB]]* [[SEC1:%.+]], [[SB]]** [[CP1]]
// CK24-DAG: store i64 {{.+}}, i64* [[S1]]
// CK24-DAG: [[SEC1]] = getelementptr {{.*}}[[SB]]* [[SEC11:%[^,]+]], i{{.+}} 0
// CK24-DAG: [[SEC11]] = load [[SB]]*, [[SB]]** [[SEC111:%[^,]+]],
// CK24-DAG: [[SEC111]] = getelementptr {{.*}}[[SC]]* [[VAR000:%.+]], i{{.+}} 0, i{{.+}} 2

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR000]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL17:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->p[:5])
  { p->a++; }

// Region 18
// CK24-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE18]]{{.+}}, {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE18]]{{.+}}, i8** null)
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SC]]**
// CK24-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
// CK24-DAG: store [[SC]]* [[VAR0:%.+]], [[SC]]** [[CBP0]]
// CK24-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SA]]* [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[10 x [[SA]]]* [[SEC000:%[^,]+]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[SEC000]] = getelementptr {{.*}}[[SB]]* [[SEC0000:%[^,]+]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[SEC0000]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL18:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->s.sa[3].a)
  { p->a++; }

// Region 19
// CK24-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 3, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[MTYPE19]]{{.+}}, i8** null)
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK24-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SC]]**
// CK24-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[SA]]***
// CK24-DAG: store [[SC]]* [[VAR0:%.+]], [[SC]]** [[CBP0]]
// CK24-DAG: store [[SA]]** [[SEC0:%.+]], [[SA]]*** [[CP0]]
// CK24-DAG: store i64 {{%.+}}, i64* [[S0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[10 x [[SA]]*]* [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[[SB]]* [[SEC000:%[^,]+]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[SEC000]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[SC]]**
// CK24-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to [[SA]]***
// CK24-DAG: store [[SC]]* [[VAR0]], [[SC]]** [[CBP1]]
// CK24-DAG: store [[SA]]** [[SEC0]], [[SA]]*** [[CP1]]
// CK24-DAG: store i64 {{.+}}, i64* [[S1]]

// CK24-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to [[SA]]***
// CK24-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to i32**
// CK24-DAG: store [[SA]]** [[SEC0]], [[SA]]*** [[CBP2]]
// CK24-DAG: store i32* [[SEC1:%.+]], i32** [[CP2]]
// CK24-DAG: [[SEC1]] = getelementptr {{.*}}[[SA]]* [[SEC11:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC11]] = load [[SA]]*, [[SA]]** [[SEC111:%[^,]+]],
// CK24-DAG: [[SEC111]] = getelementptr {{.*}}[10 x [[SA]]*]* [[SEC1111:%[^,]+]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[SEC1111]] = getelementptr {{.*}}[[SB]]* [[SEC11111:%[^,]+]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[SEC11111]] = getelementptr {{.*}}[[SC]]* [[VAR000:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR000]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL19:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->s.sp[3]->a)
  { p->a++; }

// Region 20
// CK24-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE20]]{{.+}}, i8** null)
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK24-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SC]]**
// CK24-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[SB]]***
// CK24-DAG: store [[SC]]* [[VAR0:%.+]], [[SC]]** [[CBP0]]
// CK24-DAG: store [[SB]]** [[SEC0:%.+]], [[SB]]*** [[CP0]]
// CK24-DAG: store i64 {{%.+}}, i64* [[S0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 2

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[SB]]***
// CK24-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
// CK24-DAG: store [[SB]]** [[SEC0]], [[SB]]*** [[CBP1]]
// CK24-DAG: store i32* [[SEC1:%.+]], i32** [[CP1]]
// CK24-DAG: [[SEC1]] = getelementptr {{.*}}[[SB]]* [[SEC11:%[^,]+]], i{{.+}} 0
// CK24-DAG: [[SEC11]] = load [[SB]]*, [[SB]]** [[SEC111:%[^,]+]],
// CK24-DAG: [[SEC111]] = getelementptr {{.*}}[[SC]]* [[VAR000:%.+]], i{{.+}} 0, i{{.+}} 2

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR000]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL20:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->p->a)
  { p->a++; }

// Region 21
// CK24-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 3, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[MTYPE21]]{{.+}}, i8** null)
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK24-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SC]]**
// CK24-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[SA]]***
// CK24-DAG: store [[SC]]* [[VAR0:%.+]], [[SC]]** [[CBP0]]
// CK24-DAG: store [[SA]]** [[SEC0:%.+]], [[SA]]*** [[CP0]]
// CK24-DAG: store i64 {{%.+}}, i64* [[S0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SB]]* [[SEC00:[^,]+]], i{{.+}} 0, i{{.+}} 4
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[SC]]**
// CK24-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to [[SA]]***
// CK24-DAG: store [[SC]]* [[VAR0]], [[SC]]** [[CBP1]]
// CK24-DAG: store [[SA]]** [[SEC0]], [[SA]]*** [[CP1]]
// CK24-DAG: store i64 {{.+}}, i64* [[S1]]

// CK24-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to [[SA]]***
// CK24-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to i32**
// CK24-DAG: store [[SA]]** [[SEC0]], [[SA]]*** [[CBP2]]
// CK24-DAG: store i32* [[SEC1:%.+]], i32** [[CP2]]
// CK24-DAG: store i64 {{.+}}, i64* [[S2]]
// CK24-DAG: [[SEC1]] = getelementptr {{.*}}[[SA]]* [[SEC11:%[^,]+]], i{{.+}} 0
// CK24-DAG: [[SEC11]] = load [[SA]]*, [[SA]]** [[SEC111:%[^,]+]],
// CK24-DAG: [[SEC111]] = getelementptr {{.*}}[[SB]]* [[SEC1111:[^,]+]], i{{.+}} 0, i{{.+}} 4
// CK24-DAG: [[SEC1111]] = getelementptr {{.*}}[[SC]]* [[VAR000:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR000]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL21:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->s.p->a)
  { p->a++; }

// Region 22
// CK24-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE22]]{{.+}}, i8** null)
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK24-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SC]]**
// CK24-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
// CK24-DAG: store [[SC]]* [[VAR0:%.+]], [[SC]]** [[CBP0]]
// CK24-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
// CK24-DAG: store i64 {{%.+}}, i64* [[S0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[10 x i32]* [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[[SA]]* [[SEC000:%[^,]+]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[SEC000]] = getelementptr {{.*}}[[SB]]* [[SEC0000:%[^,]+]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[SEC0000]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[SC]]**
// CK24-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
// CK24-DAG: store [[SC]]* [[VAR0]], [[SC]]** [[CBP1]]
// CK24-DAG: store i32* [[SEC0]], i32** [[CP1]]
// CK24-DAG: store i64 {{.+}}, i64* [[S1]]

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL22:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->s.s.b[:2])
  { p->a++; }

// Region 23
// CK24-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 3, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[MTYPE23]]{{.+}}, i8** null)
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK24-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SC]]**
// CK24-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[SA]]***
// CK24-DAG: store [[SC]]* [[VAR0:%.+]], [[SC]]** [[CBP0]]
// CK24-DAG: store [[SA]]** [[SEC0:%.+]], [[SA]]*** [[CP0]]
// CK24-DAG: store i64 {{%.+}}, i64* [[S0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SB]]* [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 4
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[SC]]**
// CK24-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to [[SA]]***
// CK24-DAG: store [[SC]]* [[VAR0]], [[SC]]** [[CBP1]]
// CK24-DAG: store [[SA]]** [[SEC0]], [[SA]]*** [[CP1]]
// CK24-DAG: store i64 {{.+}}, i64* [[S1]]

// CK24-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to [[SA]]***
// CK24-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to i32**
// CK24-DAG: store [[SA]]** [[SEC0]], [[SA]]*** [[CBP2]]
// CK24-DAG: store i32* [[SEC1:%.+]], i32** [[CP2]]
// CK24-DAG: store i64 {{.+}}, i64* [[S2]]
// CK24-DAG: [[SEC1]] = getelementptr {{.*}}[10 x i32]* [[SEC11:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC11]] = getelementptr {{.*}}[[SA]]* [[SEC111:%[^,]+]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[SEC111]] = load [[SA]]*, [[SA]]** [[SEC1111:%[^,]+]],
// CK24-DAG: [[SEC1111]] = getelementptr {{.*}}[[SB]]* [[SEC11111:%[^,]+]], i{{.+}} 0, i{{.+}} 4
// CK24-DAG: [[SEC11111]] = getelementptr {{.*}}[[SC]]* [[VAR000:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR000]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL23:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->s.p->b[:2])
  { p->a++; }

// Region 24
// CK24-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 4, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[4 x i{{.+}}]* [[MTYPE24]]{{.+}}, i8** null)
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK24-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SC]]**
// CK24-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[SB]]***
// CK24-DAG: store [[SC]]* [[VAR0:%.+]], [[SC]]** [[CBP0]]
// CK24-DAG: store [[SB]]** [[SEC0:%.+]], [[SB]]*** [[CP0]]
// CK24-DAG: store i64 {{%.+}}, i64* [[S0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 2

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[SB]]***
// CK24-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to [[SA]]***
// CK24-DAG: store [[SB]]** [[SEC0]], [[SB]]*** [[CBP1]]
// CK24-DAG: store [[SA]]** [[SEC1:%.+]], [[SA]]*** [[CP1]]
// CK24-DAG: store i64 {{.+}}, i64* [[S1]]
// CK24-DAG: [[SEC1]] = getelementptr {{.*}}[[SB]]* [[SEC11:%[^,]+]], i{{.+}} 0, i{{.+}} 4
// CK24-DAG: [[SEC11]] = load [[SB]]*, [[SB]]** [[SEC111:%[^,]+]],
// CK24-DAG: [[SEC111]] = getelementptr {{.*}}[[SC]]* [[VAR000:%.+]], i{{.+}} 0, i{{.+}} 2

// CK24-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to [[SA]]***
// CK24-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to [[SA]]***
// CK24-DAG: store [[SA]]** [[SEC1]], [[SA]]*** [[CBP2]]
// CK24-DAG: store [[SA]]** [[SEC2:%.+]], [[SA]]*** [[CP2]]
// CK24-DAG: store i64 {{.+}}, i64* [[S2]]
// CK24-DAG: [[SEC2]] = getelementptr {{.*}}[[SA]]* [[SEC22:%[^,]+]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[SEC22]] = load [[SA]]*, [[SA]]** [[SEC222:%[^,]+]],
// CK24-DAG: [[SEC222]] = getelementptr {{.*}}[[SB]]* [[SEC2222:%[^,]+]], i{{.+}} 0, i{{.+}} 4
// CK24-DAG: [[SEC2222]] = load [[SB]]*, [[SB]]** [[SEC22222:%[^,]+]],
// CK24-DAG: [[SEC22222]] = getelementptr {{.*}}[[SC]]* [[VAR0000:%.+]], i{{.+}} 0, i{{.+}} 2

// CK24-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[S3:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[CBP3:%.+]] = bitcast i8** [[BP3]] to [[SA]]***
// CK24-DAG: [[CP3:%.+]] = bitcast i8** [[P3]] to i32**
// CK24-DAG: store [[SA]]** [[SEC2]], [[SA]]*** [[CBP3]]
// CK24-DAG: store i32* [[SEC3:%.+]], i32** [[CP3]]
// CK24-DAG: store i64 {{.+}}, i64* [[S3]]
// CK24-DAG: [[SEC3]] = getelementptr {{.*}}[[SA]]* [[SEC33:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC33]] = load [[SA]]*, [[SA]]** [[SEC333:%[^,]+]],
// CK24-DAG: [[SEC333]] = getelementptr {{.*}}[[SA]]* [[SEC3333:%[^,]+]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[SEC3333]] = load [[SA]]*, [[SA]]** [[SEC33333:%[^,]+]],
// CK24-DAG: [[SEC33333]] = getelementptr {{.*}}[[SB]]* [[SEC333333:%[^,]+]], i{{.+}} 0, i{{.+}} 4
// CK24-DAG: [[SEC333333]] = load [[SB]]*, [[SB]]** [[SEC3333333:%[^,]+]],
// CK24-DAG: [[SEC3333333]] = getelementptr {{.*}}[[SC]]* [[VAR00000:%.+]], i{{.+}} 0, i{{.+}} 2

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR000]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR0000]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00000]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL24:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->p->p->p->a)
  { p->a++; }

  return s.a;
}

// CK24: define {{.+}}[[CALL01]]
// CK24: define {{.+}}[[CALL13]]
// CK24: define {{.+}}[[CALL14]]
// CK24: define {{.+}}[[CALL15]]
// CK24: define {{.+}}[[CALL16]]
// CK24: define {{.+}}[[CALL17]]
// CK24: define {{.+}}[[CALL18]]
// CK24: define {{.+}}[[CALL19]]
// CK24: define {{.+}}[[CALL20]]
// CK24: define {{.+}}[[CALL21]]
// CK24: define {{.+}}[[CALL22]]
// CK24: define {{.+}}[[CALL23]]
// CK24: define {{.+}}[[CALL24]]
#endif // CK24
#endif
