// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK-DAG: [[TT:%.+]] = type { i64, i8 }
// CHECK-DAG: [[S1:%.+]] = type { double }

// We have 8 target regions, but only 7 that actually will generate offloading
// code, only 6 will have mapped arguments, and only 4 have all-constant map
// sizes.

// CHECK-DAG: [[SIZET2:@.+]] = private unnamed_addr constant [1 x i{{32|64}}] [i[[SZ:32|64]] 2]
// CHECK-DAG: [[MAPT2:@.+]] = private unnamed_addr constant [1 x i32] [i32 3]
// CHECK-DAG: [[SIZET3:@.+]] = private unnamed_addr constant [2 x i[[SZ]]] [i[[SZ]] 4, i[[SZ]] 2]
// CHECK-DAG: [[MAPT3:@.+]] = private unnamed_addr constant [2 x i32] [i32 3, i32 3]
// CHECK-DAG: [[MAPT4:@.+]] = private unnamed_addr constant [9 x i32] [i32 3, i32 3, i32 1, i32 3, i32 3, i32 1, i32 1, i32 3, i32 3]
// CHECK-DAG: [[SIZET5:@.+]] = private unnamed_addr constant [3 x i[[SZ]]] [i[[SZ]] 4, i[[SZ]] 2, i[[SZ]] 40]
// CHECK-DAG: [[MAPT5:@.+]] = private unnamed_addr constant [3 x i32] [i32 3, i32 3, i32 3]
// CHECK-DAG: [[SIZET6:@.+]] = private unnamed_addr constant [4 x i[[SZ]]] [i[[SZ]] 4, i[[SZ]] 2, i[[SZ]] 1, i[[SZ]] 40]
// CHECK-DAG: [[MAPT6:@.+]] = private unnamed_addr constant [4 x i32] [i32 3, i32 3, i32 3, i32 3]
// CHECK-DAG: [[MAPT7:@.+]] = private unnamed_addr constant [5 x i32] [i32 3, i32 3, i32 1, i32 1, i32 3]
// CHECK-DAG: @{{.*}} = private constant i8 0
// CHECK-DAG: @{{.*}} = private constant i8 0
// CHECK-DAG: @{{.*}} = private constant i8 0
// CHECK-DAG: @{{.*}} = private constant i8 0
// CHECK-DAG: @{{.*}} = private constant i8 0
// CHECK-DAG: @{{.*}} = private constant i8 0
// CHECK-DAG: @{{.*}} = private constant i8 0

template<typename tx, typename ty>
struct TT{
  tx X;
  ty Y;
};

// CHECK: define {{.*}}[[FOO:@.+]](
int foo(int n) {
  int a = 0;
  short aa = 0;
  float b[10];
  float bn[n];
  double c[5][10];
  double cn[5][n];
  TT<long long, char> d;

  // CHECK:       [[RET:%.+]] = call i32 @__tgt_target(i32 -1, i8* @{{[^,]+}}, i32 0, i8** null, i8** null, i[[SZ]]* null, i32* null)
  // CHECK:       store i32 [[RET]], i32* [[RHV:%.+]], align 4
  // CHECK:       [[RET2:%.+]] = load i32, i32* [[RHV]], align 4
  // CHECK-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET2]], 0
  // CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]
  // CHECK:       [[FAIL]]
  // CHECK:       call void [[HVT0:@.+]]()
  // CHECK-NEXT:  br label %[[END]]
  // CHECK:       [[END]]
  #pragma omp target
  {
  }

  // CHECK:       store i32 0, i32* [[RHV:%.+]], align 4
  // CHECK:       store i32 -1, i32* [[RHV]], align 4
  // CHECK:       [[RET2:%.+]] = load i32, i32* [[RHV]], align 4
  // CHECK-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET2]], 0
  // CHECK:       call void [[HVT1:@.+]](i32* {{[^,]+}})
  #pragma omp target if(0)
  {
    a += 1;
  }

  // CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target(i32 -1, i8* @{{[^,]+}}, i32 1, i8** [[BP:%[^,]+]], i8** [[P:%[^,]+]], i[[SZ]]* getelementptr inbounds ([1 x i[[SZ]]], [1 x i[[SZ]]]* [[SIZET2]], i32 0, i32 0), i32* getelementptr inbounds ([1 x i32], [1 x i32]* [[MAPT2]], i32 0, i32 0))
  // CHECK-DAG:   [[BP]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BPR:%[^,]+]], i32 0, i32 0
  // CHECK-DAG:   [[P]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[PR:%[^,]+]], i32 0, i32 0
  // CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BPR]], i32 0, i32 [[IDX0:[0-9]+]]
  // CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[PR]], i32 0, i32 [[IDX0]]
  // CHECK-DAG:   store i8* [[BP0:%[^,]+]], i8** [[BPADDR0]]
  // CHECK-DAG:   store i8* [[P0:%[^,]+]], i8** [[PADDR0]]
  // CHECK-DAG:   [[BP0]] = bitcast i16* %{{.+}} to i8*
  // CHECK-DAG:   [[P0]] = bitcast i16* %{{.+}} to i8*

  // CHECK:       store i32 [[RET]], i32* [[RHV:%.+]], align 4
  // CHECK:       [[RET2:%.+]] = load i32, i32* [[RHV]], align 4
  // CHECK-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET2]], 0
  // CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]
  // CHECK:       [[FAIL]]
  // CHECK:       call void [[HVT2:@.+]](i16* {{[^,]+}})
  // CHECK-NEXT:  br label %[[END]]
  // CHECK:       [[END]]
  #pragma omp target if(1)
  {
    aa += 1;
  }

  // CHECK:       [[IF:%.+]] = icmp sgt i32 {{[^,]+}}, 10
  // CHECK:       br i1 [[IF]], label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
  // CHECK:       [[IFTHEN]]
  // CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target(i32 -1, i8* @{{[^,]+}}, i32 2, i8** [[BPR:%[^,]+]], i8** [[PR:%[^,]+]], i[[SZ]]* getelementptr inbounds ([2 x i[[SZ]]], [2 x i[[SZ]]]* [[SIZET3]], i32 0, i32 0), i32* getelementptr inbounds ([2 x i32], [2 x i32]* [[MAPT3]], i32 0, i32 0))
  // CHECK-DAG:   [[BPR]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BP:%[^,]+]], i32 0, i32 0
  // CHECK-DAG:   [[PR]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[P:%[^,]+]], i32 0, i32 0

  // CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BP]], i32 0, i32 0
  // CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[P]], i32 0, i32 0
  // CHECK-DAG:   store i8* [[BP0:%[^,]+]], i8** [[BPADDR0]]
  // CHECK-DAG:   store i8* [[P0:%[^,]+]], i8** [[PADDR0]]
  // CHECK-DAG:   [[BP0]] = bitcast i32* %{{.+}} to i8*
  // CHECK-DAG:   [[P0]] = bitcast i32* %{{.+}} to i8*

  // CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BP]], i32 0, i32 1
  // CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[P]], i32 0, i32 1
  // CHECK-DAG:   store i8* [[BP1:%[^,]+]], i8** [[BPADDR1]]
  // CHECK-DAG:   store i8* [[P1:%[^,]+]], i8** [[PADDR1]]
  // CHECK-DAG:   [[BP1]] = bitcast i16* %{{.+}} to i8*
  // CHECK-DAG:   [[P1]] = bitcast i16* %{{.+}} to i8*
  // CHECK:       store i32 [[RET]], i32* [[RHV:%.+]], align 4
  // CHECK-NEXT:  br label %[[IFEND:.+]]

  // CHECK:       [[IFELSE]]
  // CHECK:       store i32 -1, i32* [[RHV]], align 4
  // CHECK-NEXT:  br label %[[IFEND:.+]]

  // CHECK:       [[IFEND]]
  // CHECK:       [[RET2:%.+]] = load i32, i32* [[RHV]], align 4
  // CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET2]], 0
  // CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
  // CHECK:       [[FAIL]]
  // CHECK:       call void [[HVT3:@.+]]({{[^,]+}}, {{[^,]+}})
  // CHECK-NEXT:  br label %[[END]]
  // CHECK:       [[END]]
  #pragma omp target if(n>10)
  {
    a += 1;
    aa += 1;
  }

  // We capture 3 VLA sizes in this target region
  // CHECK:       store i[[SZ]] [[BNELEMSIZE:%.+]], i[[SZ]]* [[VLA0:%[^,]+]]
  // CHECK:       store i[[SZ]] 5, i[[SZ]]* [[VLA1:%[^,]+]]
  // CHECK:       store i[[SZ]] [[CNELEMSIZE1:%.+]], i[[SZ]]* [[VLA2:%[^,]+]]

  // CHECK:       [[BNSIZE:%.+]] = mul nuw i[[SZ]] [[BNELEMSIZE]], 4
  // CHECK:       [[CNELEMSIZE2:%.+]] = mul nuw i[[SZ]] 5, [[CNELEMSIZE1]]
  // CHECK:       [[CNSIZE:%.+]] = mul nuw i[[SZ]] [[CNELEMSIZE2]], 8

  // CHECK:       [[IF:%.+]] = icmp sgt i32 {{[^,]+}}, 20
  // CHECK:       br i1 [[IF]], label %[[TRY:[^,]+]], label %[[FAIL:[^,]+]]
  // CHECK:       [[TRY]]
  // CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target(i32 -1, i8* @{{[^,]+}}, i32 9, i8** [[BPR:%[^,]+]], i8** [[PR:%[^,]+]], i[[SZ]]* [[SR:%[^,]+]], i32* getelementptr inbounds ([9 x i32], [9 x i32]* [[MAPT4]], i32 0, i32 0))
  // CHECK-DAG:   [[BPR]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BP:%[^,]+]], i32 0, i32 0
  // CHECK-DAG:   [[PR]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[P:%[^,]+]], i32 0, i32 0
  // CHECK-DAG:   [[SR]] = getelementptr inbounds [9 x i[[SZ]]], [9 x i[[SZ]]]* [[S:%[^,]+]], i32 0, i32 0

  // CHECK-DAG:   [[SADDR0:%.+]] = getelementptr inbounds [9 x i[[SZ]]], [9 x i[[SZ]]]* [[S]], i32 0, i32 [[IDX0:[0-9]+]]
  // CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BP]], i32 0, i32 [[IDX0]]
  // CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[P]], i32 0, i32 [[IDX0]]
  // CHECK-DAG:   [[SADDR1:%.+]] = getelementptr inbounds [9 x i[[SZ]]], [9 x i[[SZ]]]* [[S]], i32 0, i32 [[IDX1:[0-9]+]]
  // CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BP]], i32 0, i32 [[IDX1]]
  // CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[P]], i32 0, i32 [[IDX1]]
  // CHECK-DAG:   [[SADDR2:%.+]] = getelementptr inbounds [9 x i[[SZ]]], [9 x i[[SZ]]]* [[S]], i32 0, i32 [[IDX2:[0-9]+]]
  // CHECK-DAG:   [[BPADDR2:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BP]], i32 0, i32 [[IDX2]]
  // CHECK-DAG:   [[PADDR2:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[P]], i32 0, i32 [[IDX2]]
  // CHECK-DAG:   [[SADDR3:%.+]] = getelementptr inbounds [9 x i[[SZ]]], [9 x i[[SZ]]]* [[S]], i32 0, i32 [[IDX3:[0-9]+]]
  // CHECK-DAG:   [[BPADDR3:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BP]], i32 0, i32 [[IDX3]]
  // CHECK-DAG:   [[PADDR3:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[P]], i32 0, i32 [[IDX3]]
  // CHECK-DAG:   [[SADDR4:%.+]] = getelementptr inbounds [9 x i[[SZ]]], [9 x i[[SZ]]]* [[S]], i32 0, i32 [[IDX4:[0-9]+]]
  // CHECK-DAG:   [[BPADDR4:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BP]], i32 0, i32 [[IDX4]]
  // CHECK-DAG:   [[PADDR4:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[P]], i32 0, i32 [[IDX4]]
  // CHECK-DAG:   [[SADDR5:%.+]] = getelementptr inbounds [9 x i[[SZ]]], [9 x i[[SZ]]]* [[S]], i32 0, i32 [[IDX5:[0-9]+]]
  // CHECK-DAG:   [[BPADDR5:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BP]], i32 0, i32 [[IDX5]]
  // CHECK-DAG:   [[PADDR5:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[P]], i32 0, i32 [[IDX5]]
  // CHECK-DAG:   [[SADDR6:%.+]] = getelementptr inbounds [9 x i[[SZ]]], [9 x i[[SZ]]]* [[S]], i32 0, i32 [[IDX6:[0-9]+]]
  // CHECK-DAG:   [[BPADDR6:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BP]], i32 0, i32 [[IDX6]]
  // CHECK-DAG:   [[PADDR6:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[P]], i32 0, i32 [[IDX6]]
  // CHECK-DAG:   [[SADDR7:%.+]] = getelementptr inbounds [9 x i[[SZ]]], [9 x i[[SZ]]]* [[S]], i32 0, i32 [[IDX7:[0-9]+]]
  // CHECK-DAG:   [[BPADDR7:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BP]], i32 0, i32 [[IDX7]]
  // CHECK-DAG:   [[PADDR7:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[P]], i32 0, i32 [[IDX7]]
  // CHECK-DAG:   [[SADDR8:%.+]] = getelementptr inbounds [9 x i[[SZ]]], [9 x i[[SZ]]]* [[S]], i32 0, i32 [[IDX8:[0-9]+]]
  // CHECK-DAG:   [[BPADDR8:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BP]], i32 0, i32 [[IDX8]]
  // CHECK-DAG:   [[PADDR8:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[P]], i32 0, i32 [[IDX8]]

  // The names below are not necessarily consistent with the names used for the
  // addresses above as some are repeated.
  // CHECK-DAG:   [[BP0:%[^,]+]] = bitcast i[[SZ]]* [[VLA0]] to i8*
  // CHECK-DAG:   [[P0:%[^,]+]] = bitcast i[[SZ]]* [[VLA0]] to i8*
  // CHECK-DAG:   store i8* [[BP0]], i8** {{%[^,]+}}
  // CHECK-DAG:   store i8* [[P0]], i8** {{%[^,]+}}
  // CHECK-DAG:   store i[[SZ]] {{4|8}}, i[[SZ]]* {{%[^,]+}}

  // CHECK-DAG:   [[BP1:%[^,]+]] = bitcast i[[SZ]]* [[VLA1]] to i8*
  // CHECK-DAG:   [[P1:%[^,]+]] = bitcast i[[SZ]]* [[VLA1]] to i8*
  // CHECK-DAG:   store i8* [[BP1]], i8** {{%[^,]+}}
  // CHECK-DAG:   store i8* [[P1]], i8** {{%[^,]+}}
  // CHECK-DAG:   store i[[SZ]] {{4|8}}, i[[SZ]]* {{%[^,]+}}

  // CHECK-DAG:   [[BP2:%[^,]+]] = bitcast i[[SZ]]* [[VLA2]] to i8*
  // CHECK-DAG:   [[P2:%[^,]+]] = bitcast i[[SZ]]* [[VLA2]] to i8*
  // CHECK-DAG:   store i8* [[BP2]], i8** {{%[^,]+}}
  // CHECK-DAG:   store i8* [[P2]], i8** {{%[^,]+}}
  // CHECK-DAG:   store i[[SZ]] {{4|8}}, i[[SZ]]* {{%[^,]+}}

  // CHECK-DAG:   [[BP3:%[^,]+]] = bitcast i32* %{{.+}} to i8*
  // CHECK-DAG:   [[P3:%[^,]+]] = bitcast i32* %{{.+}} to i8*
  // CHECK-DAG:   store i8* [[BP3]], i8** {{%[^,]+}}
  // CHECK-DAG:   store i8* [[P3]], i8** {{%[^,]+}}
  // CHECK-DAG:   store i[[SZ]] 4, i[[SZ]]* {{%[^,]+}}

  // CHECK-DAG:   [[BP4:%[^,]+]] = bitcast [10 x float]* %{{.+}} to i8*
  // CHECK-DAG:   [[P4:%[^,]+]] = bitcast [10 x float]* %{{.+}} to i8*
  // CHECK-DAG:   store i8* [[BP4]], i8** {{%[^,]+}}
  // CHECK-DAG:   store i8* [[P4]], i8** {{%[^,]+}}
  // CHECK-DAG:   store i[[SZ]] 40, i[[SZ]]* {{%[^,]+}}

  // CHECK-DAG:   [[BP5:%[^,]+]] = bitcast float* %{{.+}} to i8*
  // CHECK-DAG:   [[P5:%[^,]+]] = bitcast float* %{{.+}} to i8*
  // CHECK-DAG:   store i8* [[BP5]], i8** {{%[^,]+}}
  // CHECK-DAG:   store i8* [[P5]], i8** {{%[^,]+}}
  // CHECK-DAG:   store i[[SZ]] [[BNSIZE]], i[[SZ]]* {{%[^,]+}}

  // CHECK-DAG:   [[BP6:%[^,]+]] = bitcast [5 x [10 x double]]* %{{.+}} to i8*
  // CHECK-DAG:   [[P6:%[^,]+]] = bitcast [5 x [10 x double]]* %{{.+}} to i8*
  // CHECK-DAG:   store i8* [[BP6]], i8** {{%[^,]+}}
  // CHECK-DAG:   store i8* [[P6]], i8** {{%[^,]+}}
  // CHECK-DAG:   store i[[SZ]] 400, i[[SZ]]* {{%[^,]+}}

  // CHECK-DAG:   [[BP7:%[^,]+]] = bitcast double* %{{.+}} to i8*
  // CHECK-DAG:   [[P7:%[^,]+]] = bitcast double* %{{.+}} to i8*
  // CHECK-DAG:   store i8* [[BP7]], i8** {{%[^,]+}}
  // CHECK-DAG:   store i8* [[P7]], i8** {{%[^,]+}}
  // CHECK-DAG:   store i[[SZ]] [[CNSIZE]], i[[SZ]]* {{%[^,]+}}

  // CHECK-DAG:   [[BP8:%[^,]+]] = bitcast [[TT]]* %{{.+}} to i8*
  // CHECK-DAG:   [[P8:%[^,]+]] = bitcast [[TT]]* %{{.+}} to i8*
  // CHECK-DAG:   store i8* [[BP8]], i8** {{%[^,]+}}
  // CHECK-DAG:   store i8* [[P8]], i8** {{%[^,]+}}
  // CHECK-DAG:   store i[[SZ]] {{12|16}}, i[[SZ]]* {{%[^,]+}}

  // CHECK:       store i32 [[RET]], i32* [[RHV:%.+]], align 4
  // CHECK:       [[RET2:%.+]] = load i32, i32* [[RHV]], align 4
  // CHECK-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET2]], 0
  // CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]

  // CHECK:       [[FAIL]]
  // CHECK:       call void [[HVT4:@.+]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}})
  // CHECK-NEXT:  br label %[[END]]
  // CHECK:       [[END]]
  #pragma omp target if(n>20)
  {
    a += 1;
    b[2] += 1.0;
    bn[3] += 1.0;
    c[1][2] += 1.0;
    cn[1][3] += 1.0;
    d.X += 1;
    d.Y += 1;
  }

  return a;
}

// Check that the offloading functions are emitted and that the arguments are
// correct and loaded correctly for the target regions in foo().

// CHECK:       define internal void [[HVT0]]()

// CHECK:       define internal void [[HVT1]](i32* dereferenceable(4) %{{.+}})
// Create stack storage and store argument in there.
// CHECK:       [[A_ADDR:%.+]] = alloca i32*, align
// CHECK:       store i32* %{{.+}}, i32** [[A_ADDR]], align
// CHECK:       [[A_ADDR2:%.+]] = load i32*, i32** [[A_ADDR]], align
// CHECK:       load i32, i32* [[A_ADDR2]], align

// CHECK:       define internal void [[HVT2]](i16* dereferenceable(2) %{{.+}})
// Create stack storage and store argument in there.
// CHECK:       [[AA_ADDR:%.+]] = alloca i16*, align
// CHECK:       store i16* %{{.+}}, i16** [[AA_ADDR]], align
// CHECK:       [[AA_ADDR2:%.+]] = load i16*, i16** [[AA_ADDR]], align
// CHECK:       load i16, i16* [[AA_ADDR2]], align

// CHECK:       define internal void [[HVT3]]
// Create stack storage and store argument in there.
// CHECK-DAG:   [[A_ADDR:%.+]] = alloca i32*, align
// CHECK-DAG:   [[AA_ADDR:%.+]] = alloca i16*, align
// CHECK-DAG:   store i32* %{{.+}}, i32** [[A_ADDR]], align
// CHECK-DAG:   store i16* %{{.+}}, i16** [[AA_ADDR]], align
// CHECK-DAG:   [[A_ADDR2:%.+]] = load i32*, i32** [[A_ADDR]], align
// CHECK-DAG:   [[AA_ADDR2:%.+]] = load i16*, i16** [[AA_ADDR]], align
// CHECK-DAG:   load i32, i32* [[A_ADDR2]], align
// CHECK-DAG:   load i16, i16* [[AA_ADDR2]], align

// CHECK:       define internal void [[HVT4]]
// Create local storage for each capture.
// CHECK-DAG:   [[LOCAL_A:%.+]] = alloca i32*
// CHECK-DAG:   [[LOCAL_B:%.+]] = alloca [10 x float]*
// CHECK-DAG:   [[LOCAL_VLA1:%.+]] = alloca i[[SZ]]*
// CHECK-DAG:   [[LOCAL_BN:%.+]] = alloca float*
// CHECK-DAG:   [[LOCAL_C:%.+]] = alloca [5 x [10 x double]]*
// CHECK-DAG:   [[LOCAL_VLA2:%.+]] = alloca i[[SZ]]*
// CHECK-DAG:   [[LOCAL_VLA3:%.+]] = alloca i[[SZ]]*
// CHECK-DAG:   [[LOCAL_CN:%.+]] = alloca double*
// CHECK-DAG:   [[LOCAL_D:%.+]] = alloca [[TT]]*
// CHECK-DAG:   store i32* [[ARG_A:%.+]], i32** [[LOCAL_A]]
// CHECK-DAG:   store [10 x float]* [[ARG_B:%.+]], [10 x float]** [[LOCAL_B]]
// CHECK-DAG:   store i[[SZ]]* [[ARG_VLA1:%.+]], i[[SZ]]** [[LOCAL_VLA1]]
// CHECK-DAG:   store float* [[ARG_BN:%.+]], float** [[LOCAL_BN]]
// CHECK-DAG:   store [5 x [10 x double]]* [[ARG_C:%.+]], [5 x [10 x double]]** [[LOCAL_C]]
// CHECK-DAG:   store i[[SZ]]* [[ARG_VLA2:%.+]], i[[SZ]]** [[LOCAL_VLA2]]
// CHECK-DAG:   store i[[SZ]]* [[ARG_VLA3:%.+]], i[[SZ]]** [[LOCAL_VLA3]]
// CHECK-DAG:   store double* [[ARG_CN:%.+]], double** [[LOCAL_CN]]
// CHECK-DAG:   store [[TT]]* [[ARG_D:%.+]], [[TT]]** [[LOCAL_D]]

// CHECK-DAG:   [[REF_A:%.+]] = load i32*, i32** [[LOCAL_A]],
// CHECK-DAG:   [[REF_B:%.+]] = load [10 x float]*, [10 x float]** [[LOCAL_B]],
// CHECK-DAG:   [[REF_VLA1:%.+]] = load i[[SZ]]*, i[[SZ]]** [[LOCAL_VLA1]],
// CHECK-DAG:   [[VAL_VLA1:%.+]] = load i[[SZ]], i[[SZ]]* [[REF_VLA1]],
// CHECK-DAG:   [[REF_BN:%.+]] = load float*, float** [[LOCAL_BN]],
// CHECK-DAG:   [[REF_C:%.+]] = load [5 x [10 x double]]*, [5 x [10 x double]]** [[LOCAL_C]],
// CHECK-DAG:   [[REF_VLA2:%.+]] = load i[[SZ]]*, i[[SZ]]** [[LOCAL_VLA2]],
// CHECK-DAG:   [[VAL_VLA2:%.+]] = load i[[SZ]], i[[SZ]]* [[REF_VLA2]],
// CHECK-DAG:   [[REF_VLA3:%.+]] = load i[[SZ]]*, i[[SZ]]** [[LOCAL_VLA3]],
// CHECK-DAG:   [[VAL_VLA3:%.+]] = load i[[SZ]], i[[SZ]]* [[REF_VLA3]],
// CHECK-DAG:   [[REF_CN:%.+]] = load double*, double** [[LOCAL_CN]],
// CHECK-DAG:   [[REF_D:%.+]] = load [[TT]]*, [[TT]]** [[LOCAL_D]],

// Use captures.
// CHECK-DAG:   load i32, i32* [[REF_A]]
// CHECK-DAG:   getelementptr inbounds [10 x float], [10 x float]* [[REF_B]], i[[SZ]] 0, i[[SZ]] 2
// CHECK-DAG:   getelementptr inbounds float, float* [[REF_BN]], i[[SZ]] 3
// CHECK-DAG:   getelementptr inbounds [5 x [10 x double]], [5 x [10 x double]]* [[REF_C]], i[[SZ]] 0, i[[SZ]] 1
// CHECK-DAG:   getelementptr inbounds double, double* [[REF_CN]], i[[SZ]] %{{.+}}
// CHECK-DAG:   getelementptr inbounds [[TT]], [[TT]]* [[REF_D]], i32 0, i32 0

template<typename tx>
tx ftemplate(int n) {
  tx a = 0;
  short aa = 0;
  tx b[10];

  #pragma omp target if(n>40)
  {
    a += 1;
    aa += 1;
    b[2] += 1;
  }

  return a;
}

static
int fstatic(int n) {
  int a = 0;
  short aa = 0;
  char aaa = 0;
  int b[10];

  #pragma omp target if(n>50)
  {
    a += 1;
    aa += 1;
    aaa += 1;
    b[2] += 1;
  }

  return a;
}

struct S1 {
  double a;

  int r1(int n){
    int b = n+1;
    short int c[2][n];

    #pragma omp target if(n>60)
    {
      this->a = (double)b + 1.5;
      c[1][1] = ++a;
    }

    return c[1][1] + (int)b;
  }
};

// CHECK: define {{.*}}@{{.*}}bar{{.*}}
int bar(int n){
  int a = 0;

  // CHECK: call {{.*}}i32 [[FOO]](i32 {{.*}})
  a += foo(n);

  S1 S;
  // CHECK: call {{.*}}i32 [[FS1:@.+]]([[S1]]* {{.*}}, i32 {{.*}})
  a += S.r1(n);

  // CHECK: call {{.*}}i32 [[FSTATIC:@.+]](i32 {{.*}})
  a += fstatic(n);

  // CHECK: call {{.*}}i32 [[FTEMPLATE:@.+]](i32 {{.*}})
  a += ftemplate<int>(n);

  return a;
}

//
// CHECK: define {{.*}}[[FS1]]
//
// We capture 2 VLA sizes in this target region
// CHECK:       store i[[SZ]] 2, i[[SZ]]* [[VLA0:%[^,]+]]
// CHECK:       store i[[SZ]] [[CELEMSIZE1:%.+]], i[[SZ]]* [[VLA1:%[^,]+]]
// CHECK:       [[CELEMSIZE2:%.+]] = mul nuw i[[SZ]] 2, [[CELEMSIZE1]]
// CHECK:       [[CSIZE:%.+]] = mul nuw i[[SZ]] [[CELEMSIZE2]], 2

// CHECK:       [[IF:%.+]] = icmp sgt i32 {{[^,]+}}, 60
// CHECK:       br i1 [[IF]], label %[[TRY:[^,]+]], label %[[FAIL:[^,]+]]
// CHECK:       [[TRY]]
// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target(i32 -1, i8* @{{[^,]+}}, i32 5, i8** [[BPR:%[^,]+]], i8** [[PR:%[^,]+]], i[[SZ]]* [[SR:%[^,]+]], i32* getelementptr inbounds ([5 x i32], [5 x i32]* [[MAPT7]], i32 0, i32 0))
// CHECK-DAG:   [[BPR]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BP:%.+]], i32 0, i32 0
// CHECK-DAG:   [[PR]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[P:%.+]], i32 0, i32 0
// CHECK-DAG:   [[SR]] = getelementptr inbounds [5 x i[[SZ]]], [5 x i[[SZ]]]* [[S:%.+]], i32 0, i32 0
// CHECK-DAG:   [[SADDR0:%.+]] = getelementptr inbounds [5 x i[[SZ]]], [5 x i[[SZ]]]* [[S]], i32 [[IDX0:[0-9]+]]
// CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BP]], i32 [[IDX0]]
// CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[P]], i32 [[IDX0]]
// CHECK-DAG:   [[SADDR1:%.+]] = getelementptr inbounds [5 x i[[SZ]]], [5 x i[[SZ]]]* [[S]], i32 [[IDX1:[0-9]+]]
// CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BP]], i32 [[IDX1]]
// CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[P]], i32 [[IDX1]]
// CHECK-DAG:   [[SADDR2:%.+]] = getelementptr inbounds [5 x i[[SZ]]], [5 x i[[SZ]]]* [[S]], i32 [[IDX2:[0-9]+]]
// CHECK-DAG:   [[BPADDR2:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BP]], i32 [[IDX2]]
// CHECK-DAG:   [[PADDR2:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[P]], i32 [[IDX2]]
// CHECK-DAG:   [[SADDR3:%.+]] = getelementptr inbounds [5 x i[[SZ]]], [5 x i[[SZ]]]* [[S]], i32 [[IDX3:[0-9]+]]
// CHECK-DAG:   [[BPADDR3:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BP]], i32 [[IDX3]]
// CHECK-DAG:   [[PADDR3:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[P]], i32 [[IDX3]]

// The names below are not necessarily consistent with the names used for the
// addresses above as some are repeated.
// CHECK-DAG:   [[BP0:%[^,]+]] = bitcast i[[SZ]]* [[VLA0]] to i8*
// CHECK-DAG:   [[P0:%[^,]+]] = bitcast i[[SZ]]* [[VLA0]] to i8*
// CHECK-DAG:   store i8* [[BP0]], i8** {{%[^,]+}}
// CHECK-DAG:   store i8* [[P0]], i8** {{%[^,]+}}
// CHECK-DAG:   store i[[SZ]] {{4|8}}, i[[SZ]]* {{%[^,]+}}

// CHECK-DAG:   [[BP1:%[^,]+]] = bitcast i[[SZ]]* [[VLA1]] to i8*
// CHECK-DAG:   [[P1:%[^,]+]] = bitcast i[[SZ]]* [[VLA1]] to i8*
// CHECK-DAG:   store i8* [[BP1]], i8** {{%[^,]+}}
// CHECK-DAG:   store i8* [[P1]], i8** {{%[^,]+}}
// CHECK-DAG:   store i[[SZ]] {{4|8}}, i[[SZ]]* {{%[^,]+}}

// CHECK-DAG:   [[BP2:%[^,]+]] = bitcast i32* %{{.+}} to i8*
// CHECK-DAG:   [[P2:%[^,]+]] = bitcast i32* %{{.+}} to i8*
// CHECK-DAG:   store i8* [[BP2]], i8** {{%[^,]+}}
// CHECK-DAG:   store i8* [[P2]], i8** {{%[^,]+}}
// CHECK-DAG:   store i[[SZ]] 4, i[[SZ]]* {{%[^,]+}}

// CHECK-DAG:   [[BP3:%[^,]+]] = bitcast [[S1]]* %{{.+}} to i8*
// CHECK-DAG:   [[P3:%[^,]+]] = bitcast [[S1]]* %{{.+}} to i8*
// CHECK-DAG:   store i8* [[BP3]], i8** {{%[^,]+}}
// CHECK-DAG:   store i8* [[P3]], i8** {{%[^,]+}}
// CHECK-DAG:   store i[[SZ]] 8, i[[SZ]]* {{%[^,]+}}

// CHECK-DAG:   [[BP4:%[^,]+]] = bitcast i16* %{{.+}} to i8*
// CHECK-DAG:   [[P4:%[^,]+]] = bitcast i16* %{{.+}} to i8*
// CHECK-DAG:   store i8* [[BP4]], i8** {{%[^,]+}}
// CHECK-DAG:   store i8* [[P4]], i8** {{%[^,]+}}
// CHECK-DAG:   store i[[SZ]] [[CSIZE]], i[[SZ]]* {{%[^,]+}}

// CHECK:       store i32 [[RET]], i32* [[RHV:%.+]], align 4
// CHECK:       [[RET2:%.+]] = load i32, i32* [[RHV]], align 4
// CHECK-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET2]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]

// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT7:@.+]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]

//
// CHECK: define {{.*}}[[FSTATIC]]
//
// CHECK:       [[IF:%.+]] = icmp sgt i32 {{[^,]+}}, 50
// CHECK:       br i1 [[IF]], label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
// CHECK:       [[IFTHEN]]
// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target(i32 -1, i8* @{{[^,]+}}, i32 4, i8** [[BPR:%[^,]+]], i8** [[PR:%[^,]+]], i[[SZ]]* getelementptr inbounds ([4 x i[[SZ]]], [4 x i[[SZ]]]* [[SIZET6]], i32 0, i32 0), i32* getelementptr inbounds ([4 x i32], [4 x i32]* [[MAPT6]], i32 0, i32 0))
// CHECK-DAG:   [[BPR]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[BP:%.+]], i32 0, i32 0
// CHECK-DAG:   [[PR]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[P:%.+]], i32 0, i32 0

// CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[BP]], i32 0, i32 0
// CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[P]], i32 0, i32 0
// CHECK-DAG:   store i8* [[BP0:%[^,]+]], i8** [[BPADDR0]]
// CHECK-DAG:   store i8* [[P0:%[^,]+]], i8** [[PADDR0]]
// CHECK-DAG:   [[BP0]] = bitcast i32* %{{.+}} to i8*
// CHECK-DAG:   [[P0]] = bitcast i32* %{{.+}} to i8*

// CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[BP]], i32 0, i32 1
// CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[P]], i32 0, i32 1
// CHECK-DAG:   store i8* [[BP1:%[^,]+]], i8** [[BPADDR1]]
// CHECK-DAG:   store i8* [[P1:%[^,]+]], i8** [[PADDR1]]
// CHECK-DAG:   [[BP1]] = bitcast i16* %{{.+}} to i8*
// CHECK-DAG:   [[P1]] = bitcast i16* %{{.+}} to i8*

// CHECK-DAG:   [[BPADDR2:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[BP]], i32 0, i32 2
// CHECK-DAG:   [[PADDR2:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[P]], i32 0, i32 2
// CHECK-DAG:   store i8* [[BP2:%[^,]+]], i8** [[BPADDR2]]
// CHECK-DAG:   store i8* [[P2:%[^,]+]], i8** [[PADDR2]]

// CHECK-DAG:   [[BPADDR3:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[BP]], i32 0, i32 3
// CHECK-DAG:   [[PADDR3:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[P]], i32 0, i32 3
// CHECK-DAG:   store i8* [[BP3:%[^,]+]], i8** [[BPADDR3]]
// CHECK-DAG:   store i8* [[P3:%[^,]+]], i8** [[PADDR3]]
// CHECK-DAG:   [[BP3]] = bitcast [10 x i32]* %{{.+}} to i8*
// CHECK-DAG:   [[P3]] = bitcast [10 x i32]* %{{.+}} to i8*

// CHECK:       store i32 [[RET]], i32* [[RHV:%.+]], align 4
// CHECK-NEXT:  br label %[[IFEND:.+]]

// CHECK:       [[IFELSE]]
// CHECK:       store i32 -1, i32* [[RHV]], align 4
// CHECK-NEXT:  br label %[[IFEND:.+]]

// CHECK:       [[IFEND]]
// CHECK:       [[RET2:%.+]] = load i32, i32* [[RHV]], align 4
// CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET2]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT6:@.+]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]

//
// CHECK: define {{.*}}[[FTEMPLATE]]
//
// CHECK:       [[IF:%.+]] = icmp sgt i32 {{[^,]+}}, 40
// CHECK:       br i1 [[IF]], label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
// CHECK:       [[IFTHEN]]
// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target(i32 -1, i8* @{{[^,]+}}, i32 3, i8** [[BPR:%[^,]+]], i8** [[PR:%[^,]+]], i[[SZ]]* getelementptr inbounds ([3 x i[[SZ]]], [3 x i[[SZ]]]* [[SIZET5]], i32 0, i32 0), i32* getelementptr inbounds ([3 x i32], [3 x i32]* [[MAPT5]], i32 0, i32 0))
// CHECK-DAG:   [[BPR]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BP:%.+]], i32 0, i32 0
// CHECK-DAG:   [[PR]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[P:%.+]], i32 0, i32 0

// CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BP]], i32 0, i32 0
// CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[P]], i32 0, i32 0
// CHECK-DAG:   store i8* [[BP0:%[^,]+]], i8** [[BPADDR0]]
// CHECK-DAG:   store i8* [[P0:%[^,]+]], i8** [[PADDR0]]
// CHECK-DAG:   [[BP0]] = bitcast i32* %{{.+}} to i8*
// CHECK-DAG:   [[P0]] = bitcast i32* %{{.+}} to i8*

// CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BP]], i32 0, i32 1
// CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[P]], i32 0, i32 1
// CHECK-DAG:   store i8* [[BP1:%[^,]+]], i8** [[BPADDR1]]
// CHECK-DAG:   store i8* [[P1:%[^,]+]], i8** [[PADDR1]]
// CHECK-DAG:   [[BP1]] = bitcast i16* %{{.+}} to i8*
// CHECK-DAG:   [[P1]] = bitcast i16* %{{.+}} to i8*

// CHECK-DAG:   [[BPADDR2:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BP]], i32 0, i32 2
// CHECK-DAG:   [[PADDR2:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[P]], i32 0, i32 2
// CHECK-DAG:   store i8* [[BP2:%[^,]+]], i8** [[BPADDR2]]
// CHECK-DAG:   store i8* [[P2:%[^,]+]], i8** [[PADDR2]]
// CHECK-DAG:   [[BP2]] = bitcast [10 x i32]* %{{.+}} to i8*
// CHECK-DAG:   [[P2]] = bitcast [10 x i32]* %{{.+}} to i8*

// CHECK:       store i32 [[RET]], i32* [[RHV:%.+]], align 4
// CHECK-NEXT:  br label %[[IFEND:.+]]

// CHECK:       [[IFELSE]]
// CHECK:       store i32 -1, i32* [[RHV]], align 4
// CHECK-NEXT:  br label %[[IFEND:.+]]

// CHECK:       [[IFEND]]
// CHECK:       [[RET2:%.+]] = load i32, i32* [[RHV]], align 4
// CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET2]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT5:@.+]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]



// Check that the offloading functions are emitted and that the arguments are
// correct and loaded correctly for the target regions of the callees of bar().

// CHECK:       define internal void [[HVT7]]
// Create local storage for each capture.
// CHECK-DAG:   [[LOCAL_THIS:%.+]] = alloca [[S1]]*
// CHECK-DAG:   [[LOCAL_B:%.+]] = alloca i32*
// CHECK-DAG:   [[LOCAL_VLA1:%.+]] = alloca i[[SZ]]*
// CHECK-DAG:   [[LOCAL_VLA2:%.+]] = alloca i[[SZ]]*
// CHECK-DAG:   [[LOCAL_C:%.+]] = alloca i16*
// CHECK-DAG:   store [[S1]]* [[ARG_THIS:%.+]], [[S1]]** [[LOCAL_THIS]]
// CHECK-DAG:   store i32* [[ARG_B:%.+]], i32** [[LOCAL_B]]
// CHECK-DAG:   store i[[SZ]]* [[ARG_VLA1:%.+]], i[[SZ]]** [[LOCAL_VLA1]]
// CHECK-DAG:   store i[[SZ]]* [[ARG_VLA2:%.+]], i[[SZ]]** [[LOCAL_VLA2]]
// CHECK-DAG:   store i16* [[ARG_C:%.+]], i16** [[LOCAL_C]]
// Store captures in the context.
// CHECK-DAG:   [[REF_THIS:%.+]] = load [[S1]]*, [[S1]]** [[LOCAL_THIS]],
// CHECK-DAG:   [[REF_B:%.+]] = load i32*, i32** [[LOCAL_B]],
// CHECK-DAG:   [[REF_VLA1:%.+]] = load i[[SZ]]*, i[[SZ]]** [[LOCAL_VLA1]],
// CHECK-DAG:   [[VAL_VLA1:%.+]] = load i[[SZ]], i[[SZ]]* [[REF_VLA1]],
// CHECK-DAG:   [[REF_VLA2:%.+]] = load i[[SZ]]*, i[[SZ]]** [[LOCAL_VLA2]],
// CHECK-DAG:   [[VAL_VLA2:%.+]] = load i[[SZ]], i[[SZ]]* [[REF_VLA2]],
// CHECK-DAG:   [[REF_C:%.+]] = load i16*, i16** [[LOCAL_C]],
// Use captures.
// CHECK-DAG:   getelementptr inbounds [[S1]], [[S1]]* [[REF_THIS]], i32 0, i32 0
// CHECK-DAG:   load i32, i32* [[REF_B]]
// CHECK-DAG:   getelementptr inbounds i16, i16* [[REF_C]], i[[SZ]] %{{.+}}


// CHECK:       define internal void [[HVT6]]
// Create local storage for each capture.
// CHECK-DAG:   [[LOCAL_A:%.+]] = alloca i32*
// CHECK-DAG:   [[LOCAL_AA:%.+]] = alloca i16*
// CHECK-DAG:   [[LOCAL_AAA:%.+]] = alloca i8*
// CHECK-DAG:   [[LOCAL_B:%.+]] = alloca [10 x i32]*
// CHECK-DAG:   store i32* [[ARG_A:%.+]], i32** [[LOCAL_A]]
// CHECK-DAG:   store i16* [[ARG_AA:%.+]], i16** [[LOCAL_AA]]
// CHECK-DAG:   store i8* [[ARG_AAA:%.+]], i8** [[LOCAL_AAA]]
// CHECK-DAG:   store [10 x i32]* [[ARG_B:%.+]], [10 x i32]** [[LOCAL_B]]
// Store captures in the context.
// CHECK-DAG:   [[REF_A:%.+]] = load i32*, i32** [[LOCAL_A]],
// CHECK-DAG:   [[REF_AA:%.+]] = load i16*, i16** [[LOCAL_AA]],
// CHECK-DAG:   [[REF_AAA:%.+]] = load i8*, i8** [[LOCAL_AAA]],
// CHECK-DAG:   [[REF_B:%.+]] = load [10 x i32]*, [10 x i32]** [[LOCAL_B]],
// Use captures.
// CHECK-DAG:   load i32, i32* [[REF_A]]
// CHECK-DAG:   load i16, i16* [[REF_AA]]
// CHECK-DAG:   load i8, i8* [[REF_AAA]]
// CHECK-DAG:   getelementptr inbounds [10 x i32], [10 x i32]* [[REF_B]], i[[SZ]] 0, i[[SZ]] 2

// CHECK:       define internal void [[HVT5]]
// Create local storage for each capture.
// CHECK-DAG:   [[LOCAL_A:%.+]] = alloca i32*
// CHECK-DAG:   [[LOCAL_AA:%.+]] = alloca i16*
// CHECK-DAG:   [[LOCAL_B:%.+]] = alloca [10 x i32]*
// CHECK-DAG:   store i32* [[ARG_A:%.+]], i32** [[LOCAL_A]]
// CHECK-DAG:   store i16* [[ARG_AA:%.+]], i16** [[LOCAL_AA]]
// CHECK-DAG:   store [10 x i32]* [[ARG_B:%.+]], [10 x i32]** [[LOCAL_B]]
// Store captures in the context.
// CHECK-DAG:   [[REF_A:%.+]] = load i32*, i32** [[LOCAL_A]],
// CHECK-DAG:   [[REF_AA:%.+]] = load i16*, i16** [[LOCAL_AA]],
// CHECK-DAG:   [[REF_B:%.+]] = load [10 x i32]*, [10 x i32]** [[LOCAL_B]],
// Use captures.
// CHECK-DAG:   load i32, i32* [[REF_A]]
// CHECK-DAG:   load i16, i16* [[REF_AA]]
// CHECK-DAG:   getelementptr inbounds [10 x i32], [10 x i32]* [[REF_B]], i[[SZ]] 0, i[[SZ]] 2
#endif
