// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DUSE -DCK31B -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefixes=CK31B,CK31B-64,CK31B-USE
// RUN: %clang_cc1 -DUSE -DCK31B -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK31B,CK31B-64,CK31B-USE
// RUN: %clang_cc1 -DUSE -DCK31B -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK31B,CK31B-32,CK31B-USE
// RUN: %clang_cc1 -DUSE -DCK31B -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK31B,CK31B-32,CK31B-USE

// RUN: %clang_cc1 -DUSE -DCK31B -verify -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DUSE -DCK31B -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DUSE -DCK31B -verify -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DUSE -DCK31B -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s

// RUN: %clang_cc1 -DCK31B -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefixes=CK31B,CK31B-64,CK31B-NOUSE
// RUN: %clang_cc1 -DCK31B -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK31B,CK31B-64,CK31B-NOUSE
// RUN: %clang_cc1 -DCK31B -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK31B,CK31B-32,CK31B-NOUSE
// RUN: %clang_cc1 -DCK31B -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK31B,CK31B-32,CK31B-NOUSE

// RUN: %clang_cc1 -DCK31B -verify -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK31B -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK31B -verify -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK31B -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// SIMD-ONLY18-NOT: {{__kmpc|__tgt}}
#ifdef CK31B

// CK31B: [[ST:%.+]] = type { i32, i32 }

// PRESENT=0x1000 | TARGET_PARAM=0x20 = 0x1020
// MEMBER_OF_1=0x1000000000000 | FROM=0x2 | TO=0x1 = 0x1000000000003
// MEMBER_OF_1=0x1000000000000 | PRESENT=0x1000 | FROM=0x2 | TO=0x1 = 0x1000000001003

// CK31B-LABEL: @.__omp_offloading_{{.*}}test_present_members{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK31B: [[MTYPE00:@.+]] = private {{.*}}constant [3 x i64] [i64 [[#0x1020]],
// CK31B-SAME: {{^}} i64 [[#0x1000000000003]], i64 [[#0x1000000001003]]]

struct ST {
  int i;
  int j;
  // CK31B-LABEL: define {{.*}}test_present_members{{.*}}(
  void test_present_members() {
    // Make sure the struct picks up present even if another element of the
    // struct doesn't have present.
    // Region 00
    // CK31B: [[I:%.+]] = getelementptr inbounds [[ST]], [[ST]]* [[THIS:%.+]], i{{.+}} 0, i{{.+}} 0
    // CK31B: [[J:%.+]] = getelementptr inbounds [[ST]], [[ST]]* [[THIS]], i{{.+}} 0, i{{.+}} 1
    // CK31B-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 3, i8** [[GEPBP:%[0-9]+]], i8** [[GEPP:%[0-9]+]], i64* [[GEPS:%.+]], i64* getelementptr {{.+}}[3 x i{{.+}}]* [[MTYPE00]]{{.+}})
    // CK31B-DAG: [[GEPS]] = getelementptr inbounds [3 x i64], [3 x i64]* [[S:%.+]], i{{.+}} 0, i{{.+}} 0
    // CK31B-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK31B-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // this
    // CK31B-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK31B-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK31B-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
    // CK31B-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
    // CK31B-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
    // CK31B-DAG: store [[ST]]* [[THIS]], [[ST]]** [[CBP0]]
    // CK31B-DAG: store i32* [[I]], i32** [[CP0]]
    // CK31B-DAG: store i64 %{{.+}}, i64* [[S0]]

    // i
    // CK31B-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
    // CK31B-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
    // CK31B-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
    // CK31B-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[ST]]**
    // CK31B-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
    // CK31B-DAG: store [[ST]]* [[THIS]], [[ST]]** [[CBP1]]
    // CK31B-DAG: store i32* [[I]], i32** [[CP1]]
    // CK31B-DAG: store i64 4, i64* [[S1]]

    // j
    // CK31B-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
    // CK31B-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
    // CK31B-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
    // CK31B-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to [[ST]]**
    // CK31B-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to i32**
    // CK31B-DAG: store [[ST]]* [[THIS]], [[ST]]** [[CBP2]]
    // CK31B-DAG: store i32* [[J]], i32** [[CP2]]
    // CK31B-DAG: store i64 4, i64* [[S2]]

    // CK31B-USE: call void [[CALL00:@.+]]([[ST]]* [[THIS]])
    // CK31B-NOUSE: call void [[CALL00:@.+]]()
    #pragma omp target map(tofrom: i) map(present, tofrom: j)
    {
#ifdef USE
      i++;
      j++;
#endif
    }
  }
};

void test() {
  ST s;
  s.test_present_members();
}

// CK31B: define {{.+}}[[CALL00]]

#endif // CK31B
#endif
