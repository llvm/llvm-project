///==========================================================================///
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY0 %s

// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

class C {
public:
  int a;
};

#pragma omp declare mapper(id: C s) map(s.a)

// CHECK-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l49.region_id = weak constant i8 0

// CHECK-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i{{64|32}}] [i{{64|32}} 4]
// CHECK-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 35]

// CHECK-LABEL: foo{{.*}}(
void foo(int a){
  int i = a;
  C c;
  c.a = a;

  // CHECK-DAG: call i32 @__tgt_target(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}})
  // CHECK-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CHECK-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CHECK-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CHECK-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CHECK-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to %class.C**
  // CHECK-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to %class.C**
  // CHECK-DAG: store %class.C* [[VAL:%[^,]+]], %class.C** [[CBP1]]
  // CHECK-DAG: store %class.C* [[VAL]], %class.C** [[CP1]]
  #pragma omp target map(mapper(id),tofrom: c)
  {
   ++c.a;
  }
  // CHECK: call void [[KERNEL:@.+]](%class.C* [[VAL]])
}


// CHECK: define internal void [[KERNEL]](%class.C* {{.+}}[[ARG:%.+]])
// CHECK: [[ADDR:%.+]] = alloca %class.C*,
// CHECK: store %class.C* [[ARG]], %class.C** [[ADDR]]
// CHECK: [[CADDR:%.+]] = load %class.C*, %class.C** [[ADDR]]
// CHECK: [[CAADDR:%.+]] = getelementptr inbounds %class.C, %class.C* [[CADDR]], i32 0, i32 0
// CHECK: [[VAL:%[^,]+]] = load i32, i32* [[CAADDR]]
// CHECK: {{.+}} = add nsw i32 [[VAL]], 1
// CHECK: }

#endif
