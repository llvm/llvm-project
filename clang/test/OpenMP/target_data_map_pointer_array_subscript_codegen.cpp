// Test host codegen.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

#pragma omp declare target
typedef struct {
  int *arr;
} MyObject;

MyObject *objects;
#pragma omp end declare target

// CHECK-DAG: [[SIZES0:@.+]] = private unnamed_addr constant [1 x i64] [i64 {{8|4}}]
// CHECK-DAG: [[MAPS0:@.+]] = private unnamed_addr constant [1 x i64] [i64 17]
// CHECK-DAG: [[SIZES1:@.+]] = private unnamed_addr constant [2 x i64] [i64 0, i64 4]
// CHECK-DAG: [[MAPS1:@.+]] = private unnamed_addr constant [2 x i64] [i64 0, i64 281474976710673]
// CHECK: @main
int main(void) {
// CHECK: [[BPTR0:%.+]] = getelementptr inbounds [1 x ptr], ptr %{{.+}}, i32 0, i32 0
// CHECK: store ptr @objects, ptr [[BPTR0]],
// CHECK: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr %{{.+}}, ptr %{{.+}}, ptr [[SIZES0]], ptr [[MAPS0]], ptr null, ptr null)
#pragma omp target enter data map(to : objects [1:1])
// CHECK: [[OBJ:%.+]] = load ptr, ptr @objects,
// CHECK: [[BPTR0:%.+]] = getelementptr inbounds [2 x ptr], ptr %{{.+}}, i32 0, i32 0
// CHECK: store ptr [[OBJ]], ptr [[BPTR0]],
// CHECK: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 -1, i32 2, ptr %{{.+}}, ptr %{{.+}}, ptr %{{.+}}, ptr [[MAPS1]], ptr null, ptr null)
#pragma omp target enter data map(to : objects[1].arr [0:1])

  return 0;
}
#endif
