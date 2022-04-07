// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -no-opaque-pointers -DCK26 -std=c++11 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK26 --check-prefix CK26-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK26 -std=c++11 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -std=c++11 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK26 --check-prefix CK26-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK26 -std=c++11 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK26 --check-prefix CK26-32
// RUN: %clang_cc1 -no-opaque-pointers -DCK26 -std=c++11 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -std=c++11 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK26 --check-prefix CK26-32

// RUN: %clang_cc1 -no-opaque-pointers -DCK26 -std=c++11 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK26 --check-prefix CK26-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK26 -std=c++11 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=45 -std=c++11 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK26 --check-prefix CK26-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK26 -std=c++11 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK26 --check-prefix CK26-32
// RUN: %clang_cc1 -no-opaque-pointers -DCK26 -std=c++11 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=45 -std=c++11 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK26 --check-prefix CK26-32

// RUN: %clang_cc1 -no-opaque-pointers -DCK26 -std=c++11 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK26 --check-prefix CK26-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK26 -std=c++11 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=50 -std=c++11 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK26 --check-prefix CK26-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK26 -std=c++11 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK26 --check-prefix CK26-32
// RUN: %clang_cc1 -no-opaque-pointers -DCK26 -std=c++11 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=50 -std=c++11 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK26 --check-prefix CK26-32

// RUN: %clang_cc1 -no-opaque-pointers -DCK26 -std=c++11 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY25 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK26 -std=c++11 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -std=c++11 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY25 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK26 -std=c++11 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY25 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK26 -std=c++11 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -std=c++11 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY25 %s
// SIMD-ONLY25-NOT: {{__kmpc|__tgt}}
#ifdef CK26
// CK26: [[ST:%.+]] = type { i32, float*, i32, float* }

// CK26-LABEL: @.__omp_offloading_{{.*}}CC{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK26: [[SIZE00:@.+]] = private {{.*}}constant [2 x i64] [i64 {{32|16}}, i64 4]
// CK26: [[MTYPE00:@.+]] = private {{.*}}constant [2 x i64] [i64 547, i64 35]

// CK26-LABEL: @.__omp_offloading_{{.*}}CC{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK26: [[SIZE01:@.+]] = private {{.*}}constant [2 x i64] [i64 {{32|16}}, i64 4]
// CK26: [[MTYPE01:@.+]] = private {{.*}}constant [2 x i64] [i64 547, i64 35]

// CK26-LABEL: @.__omp_offloading_{{.*}}CC{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK26: [[SIZE02:@.+]] = private {{.*}}constant [2 x i64] [i64 {{32|16}}, i64 4]
// CK26: [[MTYPE02:@.+]] = private {{.*}}constant [2 x i64] [i64 547, i64 35]

// CK26-LABEL: @.__omp_offloading_{{.*}}CC{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK26: [[SIZE03:@.+]] = private {{.*}}constant [2 x i64] [i64 {{32|16}}, i64 4]
// CK26: [[MTYPE03:@.+]] = private {{.*}}constant [2 x i64] [i64 547, i64 35]

// CK26-LABEL: explicit_maps_with_private_class_members{{.*}}(

struct CC {
  int fA;
  float &fB;
  int pA;
  float &pB;

  CC(float &B) : fB(B), pB(B) {

    // CK26: call {{.*}}@__kmpc_fork_call{{.*}} [[OUTCALL:@.+]] to void (i32*, i32*, ...)*
    // define {{.*}}void [[OUTCALL]]
    #pragma omp parallel firstprivate(fA,fB) private(pA,pB)
    {
      // Region 00
      // CK26-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null, i8** null)
      // CK26-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
      // CK26-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

      // CK26-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
      // CK26-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
      // CK26-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
      // CK26-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[ST]]**
      // CK26-DAG: store [[ST]]* [[VAR0:%.+]], [[ST]]** [[CBP0]]
      // CK26-DAG: store [[ST]]* [[VAR0]], [[ST]]** [[CP0]]

      // CK26-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
      // CK26-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
      // CK26-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32**
      // CK26-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
      // CK26-DAG: store i32* [[VAR1:%.+]], i32** [[CBP1]]
      // CK26-DAG: store i32* [[SEC1:%.+]], i32** [[CP1]]
      // CK26-DAG: [[VAR1]] = load i32*, i32** [[PVT:%.+]],
      // CK26-DAG: [[SEC1]] = load i32*, i32** [[PVT]],

      // CK26: call void [[CALL00:@.+]]([[ST]]* {{[^,]+}}, i32* {{[^,]+}})
      #pragma omp target map(fA)
      {
        ++fA;
      }

      // Region 01
      // CK26-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE01]]{{.+}}, i8** null, i8** null)
      // CK26-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
      // CK26-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

      // CK26-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
      // CK26-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
      // CK26-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
      // CK26-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[ST]]**
      // CK26-DAG: store [[ST]]* [[VAR0:%.+]], [[ST]]** [[CBP0]]
      // CK26-DAG: store [[ST]]* [[VAR0]], [[ST]]** [[CP0]]

      // CK26-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
      // CK26-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
      // CK26-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to float**
      // CK26-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to float**
      // CK26-DAG: store float* [[VAR1:%.+]], float** [[CBP1]]
      // CK26-DAG: store float* [[SEC1:%.+]], float** [[CP1]]
      // CK26-DAG: [[VAR1]] = load float*, float** [[PVT:%.+]],
      // CK26-DAG: [[SEC1]] = load float*, float** [[PVT]],

      // CK26: call void [[CALL01:@.+]]([[ST]]* {{[^,]+}}, float* {{[^,]+}})
      #pragma omp target map(fB)
      {
        fB += 1.0;
      }

      // Region 02
      // CK26-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE02]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE02]]{{.+}}, i8** null, i8** null)
      // CK26-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
      // CK26-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

      // CK26-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
      // CK26-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
      // CK26-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
      // CK26-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[ST]]**
      // CK26-DAG: store [[ST]]* [[VAR0:%.+]], [[ST]]** [[CBP0]]
      // CK26-DAG: store [[ST]]* [[VAR0]], [[ST]]** [[CP0]]

      // CK26-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
      // CK26-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
      // CK26-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32**
      // CK26-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
      // CK26-DAG: store i32* [[VAR1:%.+]], i32** [[CBP1]]
      // CK26-DAG: store i32* [[SEC1:%.+]], i32** [[CP1]]
      // CK26-DAG: [[VAR1]] = load i32*, i32** [[PVT:%.+]],
      // CK26-DAG: [[SEC1]] = load i32*, i32** [[PVT]],

      // CK26: call void [[CALL02:@.+]]([[ST]]* {{[^,]+}}, i32* {{[^,]+}})
      #pragma omp target map(pA)
      {
        ++pA;
      }

      // Region 01
      // CK26-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE03]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE03]]{{.+}}, i8** null, i8** null)
      // CK26-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
      // CK26-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

      // CK26-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
      // CK26-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
      // CK26-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
      // CK26-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[ST]]**
      // CK26-DAG: store [[ST]]* [[VAR0:%.+]], [[ST]]** [[CBP0]]
      // CK26-DAG: store [[ST]]* [[VAR0]], [[ST]]** [[CP0]]

      // CK26-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
      // CK26-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
      // CK26-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to float**
      // CK26-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to float**
      // CK26-DAG: store float* [[VAR1:%.+]], float** [[CBP1]]
      // CK26-DAG: store float* [[SEC1:%.+]], float** [[CP1]]
      // CK26-DAG: [[VAR1]] = load float*, float** [[PVT:%.+]],
      // CK26-DAG: [[SEC1]] = load float*, float** [[PVT]],

      // CK26: call void [[CALL03:@.+]]([[ST]]* {{[^,]+}}, float* {{[^,]+}})
      #pragma omp target map(pB)
      {
        pB += 1.0;
      }
    }
  }

  int foo() {
    return fA + pA;
  }
};

// Make sure the private instance is used in all target regions.
// CK26: define {{.+}}[[CALL00]]({{.*}}i32*{{.*}}[[PVTARG:%.+]])
// CK26: store i32* [[PVTARG]], i32** [[PVTADDR:%.+]],
// CK26: [[ADDR:%.+]] = load i32*, i32** [[PVTADDR]],
// CK26: store i32* [[ADDR]], i32** [[PVTADDR:%.+]],
// CK26: [[ADDR:%.+]] = load i32*, i32** [[PVTADDR]],
// CK26: [[VAL:%.+]] = load i32, i32* [[ADDR]],
// CK26: add nsw i32 [[VAL]], 1

// CK26: define {{.+}}[[CALL01]]({{.*}}float*{{.*}}[[PVTARG:%.+]])
// CK26: store float* [[PVTARG]], float** [[PVTADDR:%.+]],
// CK26: [[ADDR:%.+]] = load float*, float** [[PVTADDR]],
// CK26: store float* [[ADDR]], float** [[PVTADDR:%.+]],
// CK26: [[ADDR:%.+]] = load float*, float** [[PVTADDR]],
// CK26: [[VAL:%.+]] = load float, float* [[ADDR]],
// CK26: [[EXT:%.+]] = fpext float [[VAL]] to double
// CK26: fadd double [[EXT]], 1.000000e+00

// CK26: define {{.+}}[[CALL02]]({{.*}}i32*{{.*}}[[PVTARG:%.+]])
// CK26: store i32* [[PVTARG]], i32** [[PVTADDR:%.+]],
// CK26: [[ADDR:%.+]] = load i32*, i32** [[PVTADDR]],
// CK26: store i32* [[ADDR]], i32** [[PVTADDR:%.+]],
// CK26: [[ADDR:%.+]] = load i32*, i32** [[PVTADDR]],
// CK26: [[VAL:%.+]] = load i32, i32* [[ADDR]],
// CK26: add nsw i32 [[VAL]], 1

// CK26: define {{.+}}[[CALL03]]({{.*}}float*{{.*}}[[PVTARG:%.+]])
// CK26: store float* [[PVTARG]], float** [[PVTADDR:%.+]],
// CK26: [[ADDR:%.+]] = load float*, float** [[PVTADDR]],
// CK26: store float* [[ADDR]], float** [[PVTADDR:%.+]],
// CK26: [[ADDR:%.+]] = load float*, float** [[PVTADDR]],
// CK26: [[VAL:%.+]] = load float, float* [[ADDR]],
// CK26: [[EXT:%.+]] = fpext float [[VAL]] to double
// CK26: fadd double [[EXT]], 1.000000e+00

int explicit_maps_with_private_class_members(){
  float B;
  CC c(B);
  return c.foo();
}
#endif // CK26
#endif
