// Test host codegen.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK-DAG: [[TT:%.+]] = type { i64, i8 }
// CHECK-DAG: [[S1:%.+]] = type { double }
// CHECK-DAG: [[ENTTY:%.+]] = type { i8*, i8*, i[[SZ:32|64]], i32, i32 }
// CHECK-DAG: [[DEVTY:%.+]] = type { i8*, i8*, [[ENTTY]]*, [[ENTTY]]* }
// CHECK-DAG: [[DSCTY:%.+]] = type { i32, [[DEVTY]]*, [[ENTTY]]*, [[ENTTY]]* }

// TCHECK: [[ENTTY:%.+]] = type { i8*, i8*, i{{32|64}}, i32, i32 }

// CHECK-DAG: $[[REGFN:\.omp_offloading\..+]] = comdat

// We have 8 target regions, but only 7 that actually will generate offloading
// code, only 6 will have mapped arguments, and only 4 have all-constant map
// sizes.

// CHECK-DAG: [[SIZET2:@.+]] = private unnamed_addr constant [3 x i[[SZ]]] [i[[SZ]] 2, i[[SZ]] 4, i[[SZ]] 4]
// CHECK-DAG: [[MAPT2:@.+]] = private unnamed_addr constant [3 x i64] [i64 288, i64 288, i64 288]
// CHECK-DAG: [[SIZET3:@.+]] = private unnamed_addr constant [2 x i[[SZ]]] [i[[SZ]] 4, i[[SZ]] 2]
// CHECK-DAG: [[MAPT3:@.+]] = private unnamed_addr constant [2 x i64] [i64 288, i64 288]
// CHECK-DAG: [[MAPT4:@.+]] = private unnamed_addr constant [9 x i64] [i64 288, i64 547, i64 288, i64 547, i64 547, i64 288, i64 288, i64 547, i64 547]
// CHECK-DAG: [[SIZET5:@.+]] = private unnamed_addr constant [3 x i[[SZ]]] [i[[SZ]] 4, i[[SZ]] 2, i[[SZ]] 40]
// CHECK-DAG: [[MAPT5:@.+]] = private unnamed_addr constant [3 x i64] [i64 288, i64 288, i64 547]
// CHECK-DAG: [[SIZET6:@.+]] = private unnamed_addr constant [4 x i[[SZ]]] [i[[SZ]] 4, i[[SZ]] 2, i[[SZ]] 1, i[[SZ]] 40]
// CHECK-DAG: [[MAPT6:@.+]] = private unnamed_addr constant [4 x i64] [i64 288, i64 288, i64 288, i64 547]
// CHECK-DAG: [[MAPT7:@.+]] = private unnamed_addr constant [5 x i64] [i64 547, i64 288, i64 288, i64 288, i64 547]
// CHECK-DAG: @{{.*}} = private constant i8 0
// CHECK-DAG: @{{.*}} = private constant i8 0
// CHECK-DAG: @{{.*}} = private constant i8 0
// CHECK-DAG: @{{.*}} = private constant i8 0
// CHECK-DAG: @{{.*}} = private constant i8 0
// CHECK-DAG: @{{.*}} = private constant i8 0
// CHECK-DAG: @{{.*}} = private constant i8 0

// TCHECK: @{{.+}} = constant [[ENTTY]]
// TCHECK: @{{.+}} = constant [[ENTTY]]
// TCHECK: @{{.+}} = constant [[ENTTY]]
// TCHECK: @{{.+}} = constant [[ENTTY]]
// TCHECK: @{{.+}} = constant [[ENTTY]]
// TCHECK: @{{.+}} = constant [[ENTTY]]
// TCHECK: @{{.+}} = constant [[ENTTY]]
// TCHECK-NOT: @{{.+}} = constant [[ENTTY]]

// Check if offloading descriptor is created.
// CHECK: [[ENTBEGIN:@.+]] = external constant [[ENTTY]]
// CHECK: [[ENTEND:@.+]] = external constant [[ENTTY]]
// CHECK: [[DEVBEGIN:@.+]] = external constant i8
// CHECK: [[DEVEND:@.+]] = external constant i8
// CHECK: [[IMAGES:@.+]] = internal unnamed_addr constant [1 x [[DEVTY]]] [{{.+}} { i8* [[DEVBEGIN]], i8* [[DEVEND]], [[ENTTY]]* [[ENTBEGIN]], [[ENTTY]]* [[ENTEND]] }], comdat($[[REGFN]])
// CHECK: [[DESC:@.+]] = internal constant [[DSCTY]] { i32 1, [[DEVTY]]* getelementptr inbounds ([1 x [[DEVTY]]], [1 x [[DEVTY]]]* [[IMAGES]], i32 0, i32 0), [[ENTTY]]* [[ENTBEGIN]], [[ENTTY]]* [[ENTEND]] }, comdat($[[REGFN]])

// Check target registration is registered as a Ctor.
// CHECK: appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 0, void ()* bitcast (void (i8*)* @[[REGFN]] to void ()*), i8* bitcast (void (i8*)* @[[REGFN]] to i8*) }]


template<typename tx, typename ty>
struct TT{
  tx X;
  ty Y;
};

// CHECK-LABEL: get_val
long long get_val() { return 0; }

// CHECK: define {{.*}}[[FOO:@.+]](
int foo(int n) {
  int a = 0;
  short aa = 0;
  float b[10];
  float bn[n];
  double c[5][10];
  double cn[5][n];
  TT<long long, char> d;

  // CHECK:       [[RET:%.+]] = call i32 @__tgt_target_nowait(i64 -1, i8* @{{[^,]+}}, i32 0, i8** null, i8** null, i[[SZ]]* null, i64* null)
  // CHECK-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
  // CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]
  // CHECK:       [[FAIL]]
  // CHECK:       call void [[HVT0:@.+]]()
  // CHECK-NEXT:  br label %[[END]]
  // CHECK:       [[END]]
  #pragma omp target simd nowait
  for (int i = 3; i < 32; i += 5) {
  }

  // CHECK:       call void [[HVT1:@.+]](i[[SZ]] {{[^,]+}}, i{{32|64}}{{[*]*}} {{[^)]+}})
  long long k = get_val();
  #pragma omp target simd if(target: 0) linear(k : 3)
  for (int i = 10; i > 1; i--) {
    a += 1;
  }

  // CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target(i64 -1, i8* @{{[^,]+}}, i32 3, i8** [[BP:%[^,]+]], i8** [[P:%[^,]+]], i[[SZ]]* getelementptr inbounds ([3 x i[[SZ]]], [3 x i[[SZ]]]* [[SIZET2]], i32 0, i32 0), i64* getelementptr inbounds ([3 x i64], [3 x i64]* [[MAPT2]], i32 0, i32 0))
  // CHECK-DAG:   [[BP]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BPR:%[^,]+]], i32 0, i32 0
  // CHECK-DAG:   [[P]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[PR:%[^,]+]], i32 0, i32 0
  // CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BPR]], i32 0, i32 0
  // CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[PR]], i32 0, i32 0
  // CHECK-DAG:   [[CBPADDR0:%.+]] = bitcast i8** [[BPADDR0]] to i[[SZ]]*
  // CHECK-DAG:   [[CPADDR0:%.+]] = bitcast i8** [[PADDR0]] to i[[SZ]]*
  // CHECK-DAG:   store i[[SZ]] [[VAL0:%.+]], i[[SZ]]* [[CBPADDR0]],
  // CHECK-DAG:   store i[[SZ]] [[VAL0]], i[[SZ]]* [[CPADDR0]],
  // CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BPR]], i32 0, i32 1
  // CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[PR]], i32 0, i32 1
  // CHECK-DAG:   [[CBPADDR1:%.+]] = bitcast i8** [[BPADDR1]] to i[[SZ]]*
  // CHECK-DAG:   [[CPADDR1:%.+]] = bitcast i8** [[PADDR1]] to i[[SZ]]*
  // CHECK-DAG:   store i[[SZ]] [[VAL1:%.+]], i[[SZ]]* [[CBPADDR1]],
  // CHECK-DAG:   store i[[SZ]] [[VAL1]], i[[SZ]]* [[CPADDR1]],
  // CHECK-DAG:   [[BPADDR2:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BPR]], i32 0, i32 1
  // CHECK-DAG:   [[PADDR2:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[PR]], i32 0, i32 1
  // CHECK-DAG:   [[CBPADDR2:%.+]] = bitcast i8** [[BPADDR2]] to i[[SZ]]*
  // CHECK-DAG:   [[CPADDR2:%.+]] = bitcast i8** [[PADDR2]] to i[[SZ]]*
  // CHECK-DAG:   store i[[SZ]] [[VAL2:%.+]], i[[SZ]]* [[CBPADDR2]],
  // CHECK-DAG:   store i[[SZ]] [[VAL2]], i[[SZ]]* [[CPADDR2]],

  // CHECK-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
  // CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]
  // CHECK:       [[FAIL]]
  // CHECK:       call void [[HVT2:@.+]](i[[SZ]] {{[^,]+}}, i[[SZ]] {{[^)]+}})
  // CHECK-NEXT:  br label %[[END]]
  // CHECK:       [[END]]
  int lin = 12;
  #pragma omp target simd if(target: 1) linear(lin, a : get_val())
  for (unsigned long long it = 2000; it >= 600; it-=400) {
    aa += 1;
  }

  // CHECK:       [[IF:%.+]] = icmp sgt i32 {{[^,]+}}, 10
  // CHECK:       br i1 [[IF]], label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
  // CHECK:       [[IFTHEN]]
  // CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target(i64 -1, i8* @{{[^,]+}}, i32 2, i8** [[BPR:%[^,]+]], i8** [[PR:%[^,]+]], i[[SZ]]* getelementptr inbounds ([2 x i[[SZ]]], [2 x i[[SZ]]]* [[SIZET3]], i32 0, i32 0), i64* getelementptr inbounds ([2 x i64], [2 x i64]* [[MAPT3]], i32 0, i32 0))
  // CHECK-DAG:   [[BPR]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BP:%[^,]+]], i32 0, i32 0
  // CHECK-DAG:   [[PR]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[P:%[^,]+]], i32 0, i32 0

  // CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BP]], i32 0, i32 0
  // CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[P]], i32 0, i32 0
  // CHECK-DAG:   [[CBPADDR0:%.+]] = bitcast i8** [[BPADDR0]] to i[[SZ]]*
  // CHECK-DAG:   [[CPADDR0:%.+]] = bitcast i8** [[PADDR0]] to i[[SZ]]*
  // CHECK-DAG:   store i[[SZ]] [[VAL0:%.+]], i[[SZ]]* [[CBPADDR0]],
  // CHECK-DAG:   store i[[SZ]] [[VAL0]], i[[SZ]]* [[CPADDR0]],

  // CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BP]], i32 0, i32 1
  // CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[P]], i32 0, i32 1
  // CHECK-DAG:   [[CBPADDR1:%.+]] = bitcast i8** [[BPADDR1]] to i[[SZ]]*
  // CHECK-DAG:   [[CPADDR1:%.+]] = bitcast i8** [[PADDR1]] to i[[SZ]]*
  // CHECK-DAG:   store i[[SZ]] [[VAL1:%.+]], i[[SZ]]* [[CBPADDR1]],
  // CHECK-DAG:   store i[[SZ]] [[VAL1]], i[[SZ]]* [[CPADDR1]],
  // CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
  // CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
  // CHECK:       [[FAIL]]
  // CHECK:       call void [[HVT3:@.+]]({{[^,]+}}, {{[^,]+}})
  // CHECK-NEXT:  br label %[[END]]
  // CHECK:       [[END]]
  // CHECK-NEXT:  br label %[[IFEND:.+]]
  // CHECK:       [[IFELSE]]
  // CHECK:       call void [[HVT3]]({{[^,]+}}, {{[^,]+}})
  // CHECK-NEXT:  br label %[[IFEND]]
  // CHECK:       [[IFEND]]

  #pragma omp target simd if(target: n>10)
  for (short it = 6; it <= 20; it-=-4) {
    a += 1;
    aa += 1;
  }

  // We capture 3 VLA sizes in this target region
  // CHECK-64:       [[A_VAL:%.+]] = load i32, i32* %{{.+}},
  // CHECK-64:       [[A_ADDR:%.+]] = bitcast i[[SZ]]* [[A_CADDR:%.+]] to i32*
  // CHECK-64:       store i32 [[A_VAL]], i32* [[A_ADDR]],
  // CHECK-64:       [[A_CVAL:%.+]] = load i[[SZ]], i[[SZ]]* [[A_CADDR]],

  // CHECK-32:       [[A_VAL:%.+]] = load i32, i32* %{{.+}},
  // CHECK-32:       store i32 [[A_VAL]], i32* [[A_CADDR:%.+]],
  // CHECK-32:       [[A_CVAL:%.+]] = load i[[SZ]], i[[SZ]]* [[A_CADDR]],

  // CHECK:       [[IF:%.+]] = icmp sgt i32 {{[^,]+}}, 20
  // CHECK:       br i1 [[IF]], label %[[TRY:[^,]+]], label %[[FAIL:[^,]+]]
  // CHECK:       [[TRY]]
  // CHECK:       [[BNSIZE:%.+]] = mul nuw i[[SZ]] [[VLA0:%.+]], 4
  // CHECK:       [[CNELEMSIZE2:%.+]] = mul nuw i[[SZ]] 5, [[VLA1:%.+]]
  // CHECK:       [[CNSIZE:%.+]] = mul nuw i[[SZ]] [[CNELEMSIZE2]], 8

  // CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target(i64 -1, i8* @{{[^,]+}}, i32 9, i8** [[BPR:%[^,]+]], i8** [[PR:%[^,]+]], i[[SZ]]* [[SR:%[^,]+]], i64* getelementptr inbounds ([9 x i64], [9 x i64]* [[MAPT4]], i32 0, i32 0))
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
  // CHECK-DAG:   store i[[SZ]] [[VLA0]], i[[SZ]]* [[CBPADDR0:%.+]],
  // CHECK-DAG:   store i[[SZ]] [[VLA0]], i[[SZ]]* [[CPADDR0:%.+]],
  // CHECK-DAG:   [[CBPADDR0]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
  // CHECK-DAG:   [[CPADDR0]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
  // CHECK-DAG:   store i[[SZ]] {{4|8}}, i[[SZ]]* {{%[^,]+}}

  // CHECK-DAG:   store i[[SZ]] [[VLA1]], i[[SZ]]* [[CBPADDR1:%.+]],
  // CHECK-DAG:   store i[[SZ]] [[VLA1]], i[[SZ]]* [[CPADDR1:%.+]],
  // CHECK-DAG:   [[CBPADDR1]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
  // CHECK-DAG:   [[CPADDR1]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
  // CHECK-DAG:   store i[[SZ]] {{4|8}}, i[[SZ]]* {{%[^,]+}}

  // CHECK-DAG:   store i[[SZ]] 5, i[[SZ]]* [[CBPADDR2:%.+]],
  // CHECK-DAG:   store i[[SZ]] 5, i[[SZ]]* [[CPADDR2:%.+]],
  // CHECK-DAG:   [[CBPADDR2]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
  // CHECK-DAG:   [[CPADDR2]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
  // CHECK-DAG:   store i[[SZ]] {{4|8}}, i[[SZ]]* {{%[^,]+}}

  // CHECK-DAG:   store i[[SZ]] [[A_CVAL]], i[[SZ]]* [[CBPADDR3:%.+]],
  // CHECK-DAG:   store i[[SZ]] [[A_CVAL]], i[[SZ]]* [[CPADDR3:%.+]],
  // CHECK-DAG:   [[CBPADDR3]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
  // CHECK-DAG:   [[CPADDR3]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
  // CHECK-DAG:   store i[[SZ]] 4, i[[SZ]]* {{%[^,]+}}

  // CHECK-DAG:   store [10 x float]* %{{.+}}, [10 x float]** [[CBPADDR4:%.+]],
  // CHECK-DAG:   store [10 x float]* %{{.+}}, [10 x float]** [[CPADDR4:%.+]],
  // CHECK-DAG:   [[CBPADDR4]] = bitcast i8** {{%[^,]+}} to [10 x float]**
  // CHECK-DAG:   [[CPADDR4]] = bitcast i8** {{%[^,]+}} to [10 x float]**
  // CHECK-DAG:   store i[[SZ]] 40, i[[SZ]]* {{%[^,]+}}

  // CHECK-DAG:   store float* %{{.+}}, float** [[CBPADDR5:%.+]],
  // CHECK-DAG:   store float* %{{.+}}, float** [[CPADDR5:%.+]],
  // CHECK-DAG:   [[CBPADDR5]] = bitcast i8** {{%[^,]+}} to float**
  // CHECK-DAG:   [[CPADDR5]] = bitcast i8** {{%[^,]+}} to float**
  // CHECK-DAG:   store i[[SZ]] [[BNSIZE]], i[[SZ]]* {{%[^,]+}}

  // CHECK-DAG:   store [5 x [10 x double]]* %{{.+}}, [5 x [10 x double]]** [[CBPADDR6:%.+]],
  // CHECK-DAG:   store [5 x [10 x double]]* %{{.+}}, [5 x [10 x double]]** [[CPADDR6:%.+]],
  // CHECK-DAG:   [[CBPADDR6]] = bitcast i8** {{%[^,]+}} to [5 x [10 x double]]**
  // CHECK-DAG:   [[CPADDR6]] = bitcast i8** {{%[^,]+}} to [5 x [10 x double]]**
  // CHECK-DAG:   store i[[SZ]] 400, i[[SZ]]* {{%[^,]+}}

  // CHECK-DAG:   store double* %{{.+}}, double** [[CBPADDR7:%.+]],
  // CHECK-DAG:   store double* %{{.+}}, double** [[CPADDR7:%.+]],
  // CHECK-DAG:   [[CBPADDR7]] = bitcast i8** {{%[^,]+}} to double**
  // CHECK-DAG:   [[CPADDR7]] = bitcast i8** {{%[^,]+}} to double**
  // CHECK-DAG:   store i[[SZ]] [[CNSIZE]], i[[SZ]]* {{%[^,]+}}

  // CHECK-DAG:   store [[TT]]* %{{.+}}, [[TT]]** [[CBPADDR8:%.+]],
  // CHECK-DAG:   store [[TT]]* %{{.+}}, [[TT]]** [[CPADDR8:%.+]],
  // CHECK-DAG:   [[CBPADDR8]] = bitcast i8** {{%[^,]+}} to [[TT]]**
  // CHECK-DAG:   [[CPADDR8]] = bitcast i8** {{%[^,]+}} to [[TT]]**
  // CHECK-DAG:   store i[[SZ]] {{12|16}}, i[[SZ]]* {{%[^,]+}}

  // CHECK-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
  // CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]

  // CHECK:       [[FAIL]]
  // CHECK:       call void [[HVT4:@.+]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}})
  // CHECK-NEXT:  br label %[[END]]
  // CHECK:       [[END]]
  #pragma omp target simd if(target: n>20)
  for (unsigned char it = 'z'; it >= 'a'; it+=-1) {
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
// CHECK:       !llvm.loop
// CHECK:       ret void
// CHECK-NEXT:  }


// CHECK:       define internal void [[HVT1]](i[[SZ]] %{{.+}}, i{{32|64}}{{[*]*.*}} %{{.+}})
// CHECK:       [[AA_ADDR:%.+]] = alloca i[[SZ]], align
// CHECK:       store i[[SZ]] %{{.+}}, i[[SZ]]* [[AA_ADDR]], align
// CHECK-64:    [[AA_CADDR:%.+]] = bitcast i[[SZ]]* [[AA_ADDR]] to i32*
// CHECK-64:    [[AA:%.+]] = load i32, i32* [[AA_CADDR]], align
// CHECK-32:    [[AA:%.+]] = load i32, i32* [[AA_ADDR]], align
// CHECK:       !llvm.mem.parallel_loop_access
// CHECK:       !llvm.loop
// CHECK:       ret void
// CHECK-NEXT:  }

// CHECK:       define internal void [[HVT2]](i[[SZ]] %{{.+}}, i[[SZ]] %{{.+}}, i[[SZ]] %{{.+}})
// CHECK:       [[AA_ADDR:%.+]] = alloca i[[SZ]], align
// CHECK:       store i[[SZ]] %{{.+}}, i[[SZ]]* [[AA_ADDR]], align
// CHECK:       [[AA_CADDR:%.+]] = bitcast i[[SZ]]* [[AA_ADDR]] to i16*
// CHECK:       [[AA:%.+]] = load i16, i16* [[AA_CADDR]], align
// CHECK:       !llvm.loop
// CHECK:       ret void
// CHECK-NEXT:  }

// CHECK:       define internal void [[HVT3]]
// CHECK:       [[A_ADDR:%.+]] = alloca i[[SZ]], align
// CHECK:       [[AA_ADDR:%.+]] = alloca i[[SZ]], align
// CHECK-DAG:   store i[[SZ]] %{{.+}}, i[[SZ]]* [[A_ADDR]], align
// CHECK-DAG:   store i[[SZ]] %{{.+}}, i[[SZ]]* [[AA_ADDR]], align
// CHECK-64-DAG:[[A_CADDR:%.+]] = bitcast i[[SZ]]* [[A_ADDR]] to i32*
// CHECK-DAG:   [[AA_CADDR:%.+]] = bitcast i[[SZ]]* [[AA_ADDR]] to i16*
// CHECK:       !llvm.loop
// CHECK:       ret void
// CHECK-NEXT:  }

// CHECK:       define internal void [[HVT4]]
// Create local storage for each capture.
// CHECK:       [[LOCAL_A:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_B:%.+]] = alloca [10 x float]*
// CHECK:       [[LOCAL_VLA1:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_BN:%.+]] = alloca float*
// CHECK:       [[LOCAL_C:%.+]] = alloca [5 x [10 x double]]*
// CHECK:       [[LOCAL_VLA2:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_VLA3:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_CN:%.+]] = alloca double*
// CHECK:       [[LOCAL_D:%.+]] = alloca [[TT]]*
// CHECK-DAG:   store i[[SZ]] [[ARG_A:%.+]], i[[SZ]]* [[LOCAL_A]]
// CHECK-DAG:   store [10 x float]* [[ARG_B:%.+]], [10 x float]** [[LOCAL_B]]
// CHECK-DAG:   store i[[SZ]] [[ARG_VLA1:%.+]], i[[SZ]]* [[LOCAL_VLA1]]
// CHECK-DAG:   store float* [[ARG_BN:%.+]], float** [[LOCAL_BN]]
// CHECK-DAG:   store [5 x [10 x double]]* [[ARG_C:%.+]], [5 x [10 x double]]** [[LOCAL_C]]
// CHECK-DAG:   store i[[SZ]] [[ARG_VLA2:%.+]], i[[SZ]]* [[LOCAL_VLA2]]
// CHECK-DAG:   store i[[SZ]] [[ARG_VLA3:%.+]], i[[SZ]]* [[LOCAL_VLA3]]
// CHECK-DAG:   store double* [[ARG_CN:%.+]], double** [[LOCAL_CN]]
// CHECK-DAG:   store [[TT]]* [[ARG_D:%.+]], [[TT]]** [[LOCAL_D]]

// CHECK-64-DAG:[[CONV_AP:%.+]] = bitcast i[[SZ]]* [[LOCAL_A]] to i32*
// CHECK-DAG:   [[REF_B:%.+]] = load [10 x float]*, [10 x float]** [[LOCAL_B]],
// CHECK-DAG:   [[VAL_VLA1:%.+]] = load i[[SZ]], i[[SZ]]* [[LOCAL_VLA1]],
// CHECK-DAG:   [[REF_BN:%.+]] = load float*, float** [[LOCAL_BN]],
// CHECK-DAG:   [[REF_C:%.+]] = load [5 x [10 x double]]*, [5 x [10 x double]]** [[LOCAL_C]],
// CHECK-DAG:   [[VAL_VLA2:%.+]] = load i[[SZ]], i[[SZ]]* [[LOCAL_VLA2]],
// CHECK-DAG:   [[VAL_VLA3:%.+]] = load i[[SZ]], i[[SZ]]* [[LOCAL_VLA3]],
// CHECK-DAG:   [[REF_CN:%.+]] = load double*, double** [[LOCAL_CN]],
// CHECK-DAG:   [[REF_D:%.+]] = load [[TT]]*, [[TT]]** [[LOCAL_D]],


template<typename tx>
tx ftemplate(int n) {
  tx a = 0;
  short aa = 0;
  tx b[10];

  #pragma omp target simd if(target: n>40)
  for (long long i = -10; i < 10; i += 3) {
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

  #pragma omp target simd if(target: n>50)
  for (unsigned i=100; i<10; i+=10) {
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

    #pragma omp target simd if(target: n>60)
    for (unsigned long long it = 2000; it >= 600; it -= 400) {
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
// CHECK:          i8* @llvm.stacksave()
// CHECK-64:       [[B_ADDR:%.+]] = bitcast i[[SZ]]* [[B_CADDR:%.+]] to i32*
// CHECK-64:       store i32 %{{.+}}, i32* [[B_ADDR]],
// CHECK-64:       [[B_CVAL:%.+]] = load i[[SZ]], i[[SZ]]* [[B_CADDR]],

// CHECK-32:       store i32 %{{.+}}, i32* %vla_expr
// CHECK-32:       store i32 %{{.+}}, i32* [[B_ADDR:%.+]],
// CHECK-32:       [[B_CVAL:%.+]] = load i[[SZ]], i[[SZ]]* [[B_ADDR]],

// CHECK:       [[IF:%.+]] = icmp sgt i32 {{[^,]+}}, 60
// CHECK:       br i1 [[IF]], label %[[TRY:[^,]+]], label %[[FAIL:[^,]+]]
// CHECK:       [[TRY]]
// We capture 2 VLA sizes in this target region
// CHECK:       [[CELEMSIZE2:%.+]] = mul nuw i[[SZ]] 2, [[VLA0:%.+]]
// CHECK:       [[CSIZE:%.+]] = mul nuw i[[SZ]] [[CELEMSIZE2]], 2

// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target(i64 -1, i8* @{{[^,]+}}, i32 5, i8** [[BPR:%[^,]+]], i8** [[PR:%[^,]+]], i[[SZ]]* [[SR:%[^,]+]], i64* getelementptr inbounds ([5 x i64], [5 x i64]* [[MAPT7]], i32 0, i32 0))
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
// CHECK-DAG:   store i[[SZ]] [[VLA0]], i[[SZ]]* [[CBPADDR0:%.+]],
// CHECK-DAG:   store i[[SZ]] [[VLA0]], i[[SZ]]* [[CPADDR0:%.+]],
// CHECK-DAG:   [[CBPADDR0]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
// CHECK-DAG:   [[CPADDR0]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
// CHECK-DAG:   store i[[SZ]] {{4|8}}, i[[SZ]]* {{%[^,]+}}

// CHECK-DAG:   store i[[SZ]] 2, i[[SZ]]* [[CBPADDR1:%.+]],
// CHECK-DAG:   store i[[SZ]] 2, i[[SZ]]* [[CPADDR1:%.+]],
// CHECK-DAG:   [[CBPADDR1]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
// CHECK-DAG:   [[CPADDR1]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
// CHECK-DAG:   store i[[SZ]] {{4|8}}, i[[SZ]]* {{%[^,]+}}

// CHECK-DAG:   store i[[SZ]] [[B_CVAL]], i[[SZ]]* [[CBPADDR2:%.+]],
// CHECK-DAG:   store i[[SZ]] [[B_CVAL]], i[[SZ]]* [[CPADDR2:%.+]],
// CHECK-DAG:   [[CBPADDR2]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
// CHECK-DAG:   [[CPADDR2]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
// CHECK-DAG:   store i[[SZ]] 4, i[[SZ]]* {{%[^,]+}}

// CHECK-DAG:   store [[S1]]* %{{.+}}, [[S1]]** [[CBPADDR3:%.+]],
// CHECK-DAG:   store [[S1]]* %{{.+}}, [[S1]]** [[CPADDR3:%.+]],
// CHECK-DAG:   [[CBPADDR3]] = bitcast i8** {{%[^,]+}} to [[S1]]**
// CHECK-DAG:   [[CPADDR3]] = bitcast i8** {{%[^,]+}} to [[S1]]**
// CHECK-DAG:   store i[[SZ]] 8, i[[SZ]]* {{%[^,]+}}

// CHECK-DAG:   store i16* %{{.+}}, i16** [[CBPADDR4:%.+]],
// CHECK-DAG:   store i16* %{{.+}}, i16** [[CPADDR4:%.+]],
// CHECK-DAG:   [[CBPADDR4]] = bitcast i8** {{%[^,]+}} to i16**
// CHECK-DAG:   [[CPADDR4]] = bitcast i8** {{%[^,]+}} to i16**
// CHECK-DAG:   store i[[SZ]] [[CSIZE]], i[[SZ]]* {{%[^,]+}}

// CHECK-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
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
// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target(i64 -1, i8* @{{[^,]+}}, i32 4, i8** [[BPR:%[^,]+]], i8** [[PR:%[^,]+]], i[[SZ]]* getelementptr inbounds ([4 x i[[SZ]]], [4 x i[[SZ]]]* [[SIZET6]], i32 0, i32 0), i64* getelementptr inbounds ([4 x i64], [4 x i64]* [[MAPT6]], i32 0, i32 0))
// CHECK-DAG:   [[BPR]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[BP:%.+]], i32 0, i32 0
// CHECK-DAG:   [[PR]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[P:%.+]], i32 0, i32 0

// CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[BP]], i32 0, i32 0
// CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[P]], i32 0, i32 0
// CHECK-DAG:   [[CBPADDR0:%.+]] = bitcast i8** [[BPADDR0]] to i[[SZ]]*
// CHECK-DAG:   [[CPADDR0:%.+]] = bitcast i8** [[PADDR0]] to i[[SZ]]*
// CHECK-DAG:   store i[[SZ]] [[VAL0:%.+]], i[[SZ]]* [[CBPADDR0]],
// CHECK-DAG:   store i[[SZ]] [[VAL0]], i[[SZ]]* [[CPADDR0]],

// CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[BP]], i32 0, i32 1
// CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[P]], i32 0, i32 1
// CHECK-DAG:   [[CBPADDR1:%.+]] = bitcast i8** [[BPADDR1]] to i[[SZ]]*
// CHECK-DAG:   [[CPADDR1:%.+]] = bitcast i8** [[PADDR1]] to i[[SZ]]*
// CHECK-DAG:   store i[[SZ]] [[VAL1:%.+]], i[[SZ]]* [[CBPADDR1]],
// CHECK-DAG:   store i[[SZ]] [[VAL1]], i[[SZ]]* [[CPADDR1]],

// CHECK-DAG:   [[BPADDR2:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[BP]], i32 0, i32 2
// CHECK-DAG:   [[PADDR2:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[P]], i32 0, i32 2
// CHECK-DAG:   [[CBPADDR2:%.+]] = bitcast i8** [[BPADDR2]] to i[[SZ]]*
// CHECK-DAG:   [[CPADDR2:%.+]] = bitcast i8** [[PADDR2]] to i[[SZ]]*
// CHECK-DAG:   store i[[SZ]] [[VAL2:%.+]], i[[SZ]]* [[CBPADDR2]],
// CHECK-DAG:   store i[[SZ]] [[VAL2]], i[[SZ]]* [[CPADDR2]],

// CHECK-DAG:   [[BPADDR3:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[BP]], i32 0, i32 3
// CHECK-DAG:   [[PADDR3:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[P]], i32 0, i32 3
// CHECK-DAG:   [[CBPADDR3:%.+]] = bitcast i8** [[BPADDR3]] to [10 x i32]**
// CHECK-DAG:   [[CPADDR3:%.+]] = bitcast i8** [[PADDR3]] to [10 x i32]**
// CHECK-DAG:   store [10 x i32]* [[VAL3:%.+]], [10 x i32]** [[CBPADDR3]],
// CHECK-DAG:   store [10 x i32]* [[VAL3]], [10 x i32]** [[CPADDR3]],

// CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT6:@.+]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]
// CHECK-NEXT:  br label %[[IFEND:.+]]
// CHECK:       [[IFELSE]]
// CHECK:       call void [[HVT6]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK-NEXT:  br label %[[IFEND]]
// CHECK:       [[IFEND]]

//
// CHECK: define {{.*}}[[FTEMPLATE]]
//
// CHECK:       [[IF:%.+]] = icmp sgt i32 {{[^,]+}}, 40
// CHECK:       br i1 [[IF]], label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
// CHECK:       [[IFTHEN]]
// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target(i64 -1, i8* @{{[^,]+}}, i32 3, i8** [[BPR:%[^,]+]], i8** [[PR:%[^,]+]], i[[SZ]]* getelementptr inbounds ([3 x i[[SZ]]], [3 x i[[SZ]]]* [[SIZET5]], i32 0, i32 0), i64* getelementptr inbounds ([3 x i64], [3 x i64]* [[MAPT5]], i32 0, i32 0))
// CHECK-DAG:   [[BPR]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BP:%.+]], i32 0, i32 0
// CHECK-DAG:   [[PR]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[P:%.+]], i32 0, i32 0

// CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BP]], i32 0, i32 0
// CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[P]], i32 0, i32 0
// CHECK-DAG:   [[CBPADDR0:%.+]] = bitcast i8** [[BPADDR0]] to i[[SZ]]*
// CHECK-DAG:   [[CPADDR0:%.+]] = bitcast i8** [[PADDR0]] to i[[SZ]]*
// CHECK-DAG:   store i[[SZ]] [[VAL0:%.+]], i[[SZ]]* [[CBPADDR0]],
// CHECK-DAG:   store i[[SZ]] [[VAL0]], i[[SZ]]* [[CPADDR0]],

// CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BP]], i32 0, i32 1
// CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[P]], i32 0, i32 1
// CHECK-DAG:   [[CBPADDR1:%.+]] = bitcast i8** [[BPADDR1]] to i[[SZ]]*
// CHECK-DAG:   [[CPADDR1:%.+]] = bitcast i8** [[PADDR1]] to i[[SZ]]*
// CHECK-DAG:   store i[[SZ]] [[VAL1:%.+]], i[[SZ]]* [[CBPADDR1]],
// CHECK-DAG:   store i[[SZ]] [[VAL1]], i[[SZ]]* [[CPADDR1]],

// CHECK-DAG:   [[BPADDR2:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BP]], i32 0, i32 2
// CHECK-DAG:   [[PADDR2:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[P]], i32 0, i32 2
// CHECK-DAG:   [[CBPADDR2:%.+]] = bitcast i8** [[BPADDR2]] to [10 x i32]**
// CHECK-DAG:   [[CPADDR2:%.+]] = bitcast i8** [[PADDR2]] to [10 x i32]**
// CHECK-DAG:   store [10 x i32]* [[VAL2:%.+]], [10 x i32]** [[CBPADDR2]],
// CHECK-DAG:   store [10 x i32]* [[VAL2]], [10 x i32]** [[CPADDR2]],

// CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT5:@.+]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]
// CHECK-NEXT:  br label %[[IFEND:.+]]
// CHECK:       [[IFELSE]]
// CHECK:       call void [[HVT:@.+]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK-NEXT:  br label %[[IFEND]]
// CHECK:       [[IFEND]]

// Check that the offloading functions are emitted and that the arguments are
// correct and loaded correctly for the target regions of the callees of bar().

// CHECK:       define internal void [[HVT7]]
// Create local storage for each capture.
// CHECK:       [[LOCAL_THIS:%.+]] = alloca [[S1]]*
// CHECK:       [[LOCAL_B:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_VLA1:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_VLA2:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_C:%.+]] = alloca i16*
// CHECK-DAG:   store [[S1]]* [[ARG_THIS:%.+]], [[S1]]** [[LOCAL_THIS]]
// CHECK-DAG:   store i[[SZ]] [[ARG_B:%.+]], i[[SZ]]* [[LOCAL_B]]
// CHECK-DAG:   store i[[SZ]] [[ARG_VLA1:%.+]], i[[SZ]]* [[LOCAL_VLA1]]
// CHECK-DAG:   store i[[SZ]] [[ARG_VLA2:%.+]], i[[SZ]]* [[LOCAL_VLA2]]
// CHECK-DAG:   store i16* [[ARG_C:%.+]], i16** [[LOCAL_C]]
// Store captures in the context.
// CHECK-DAG:   [[REF_THIS:%.+]] = load [[S1]]*, [[S1]]** [[LOCAL_THIS]],
// CHECK-64-DAG:[[CONV_BP:%.+]] = bitcast i[[SZ]]* [[LOCAL_B]] to i32*
// CHECK-DAG:   [[VAL_VLA1:%.+]] = load i[[SZ]], i[[SZ]]* [[LOCAL_VLA1]],
// CHECK-DAG:   [[VAL_VLA2:%.+]] = load i[[SZ]], i[[SZ]]* [[LOCAL_VLA2]],
// CHECK-DAG:   [[REF_C:%.+]] = load i16*, i16** [[LOCAL_C]],


// CHECK:       define internal void [[HVT6]]
// Create local storage for each capture.
// CHECK:       [[LOCAL_A:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_AA:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_AAA:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_B:%.+]] = alloca [10 x i32]*
// CHECK-DAG:   store i[[SZ]] [[ARG_A:%.+]], i[[SZ]]* [[LOCAL_A]]
// CHECK-DAG:   store i[[SZ]] [[ARG_AA:%.+]], i[[SZ]]* [[LOCAL_AA]]
// CHECK-DAG:   store i[[SZ]] [[ARG_AAA:%.+]], i[[SZ]]* [[LOCAL_AAA]]
// CHECK-DAG:   store [10 x i32]* [[ARG_B:%.+]], [10 x i32]** [[LOCAL_B]]
// Store captures in the context.
// CHECK-64-DAG:[[CONV_AP:%.+]] = bitcast i[[SZ]]* [[LOCAL_A]] to i32*
// CHECK-DAG:   [[CONV_AAP:%.+]] = bitcast i[[SZ]]* [[LOCAL_AA]] to i16*
// CHECK-DAG:   [[CONV_AAAP:%.+]] = bitcast i[[SZ]]* [[LOCAL_AAA]] to i8*
// CHECK-DAG:   [[REF_B:%.+]] = load [10 x i32]*, [10 x i32]** [[LOCAL_B]],

// CHECK:       define internal void [[HVT5]]
// Create local storage for each capture.
// CHECK:       [[LOCAL_A:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_AA:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_B:%.+]] = alloca [10 x i32]*
// CHECK-DAG:   store i[[SZ]] [[ARG_A:%.+]], i[[SZ]]* [[LOCAL_A]]
// CHECK-DAG:   store i[[SZ]] [[ARG_AA:%.+]], i[[SZ]]* [[LOCAL_AA]]
// CHECK-DAG:   store [10 x i32]* [[ARG_B:%.+]], [10 x i32]** [[LOCAL_B]]
// Store captures in the context.
// CHECK-64-DAG:[[CONV_AP:%.+]] = bitcast i[[SZ]]* [[LOCAL_A]] to i32*
// CHECK-DAG:   [[CONV_AAP:%.+]] = bitcast i[[SZ]]* [[LOCAL_AA]] to i16*
// CHECK-DAG:   [[REF_B:%.+]] = load [10 x i32]*, [10 x i32]** [[LOCAL_B]],

#endif
