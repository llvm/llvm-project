// RUN: %clang_cc1 -std=c++98 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++11 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++14 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++17 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++20 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++23 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++2c %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK

namespace cwg438 { // cwg438: 2.7

void f() {
  long A[2];
  A[0] = 0;
  A[A[0]] = 1;
}

} // namespace cwg438

// CHECK-LABEL: define {{.*}} void @cwg438::f()()
// CHECK:         [[A:%.+]] = alloca [2 x i64]
// CHECK:         {{.+}} = getelementptr inbounds [2 x i64], ptr [[A]], i64 0, i64 0
// CHECK:         [[ARRAYIDX1:%.+]] = getelementptr inbounds [2 x i64], ptr [[A]], i64 0, i64 0
// CHECK-NEXT:    [[TEMPIDX:%.+]] = load i64, ptr [[ARRAYIDX1]]
// CHECK-NEXT:    [[ARRAYIDX2:%.+]] = getelementptr inbounds [2 x i64], ptr [[A]], i64 0, i64 [[TEMPIDX]]
// CHECK-LABEL: }
