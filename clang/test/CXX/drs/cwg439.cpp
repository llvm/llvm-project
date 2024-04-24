// RUN: %clang_cc1 -std=c++98 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++11 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++14 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++17 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++20 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++23 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++2c %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK

namespace cwg439 { // cwg439: 2.7

void f() {
  int* p1 = new int;
  const int* p2 = static_cast<const int*>(static_cast<void *>(p1));
  bool b = p1 == p2; // b will have the value true.
}

} // namespace cwg439

// We're checking that p2 was copied from p1, and then was carried over
// to the comparison without change.

// CHECK-LABEL: define {{.*}} void @cwg439::f()()
// CHECK:         [[P1:%.+]] = alloca ptr, align 8
// CHECK-NEXT:    [[P2:%.+]] = alloca ptr, align 8
// CHECK:         [[TEMP0:%.+]] = load ptr, ptr [[P1]]
// CHECK-NEXT:    store ptr [[TEMP0:%.+]], ptr [[P2]]
// CHECK-NEXT:    [[TEMP1:%.+]] = load ptr, ptr [[P1]]
// CHECK-NEXT:    [[TEMP2:%.+]] = load ptr, ptr [[P2]]
// CHECK-NEXT:    {{.*}} = icmp eq ptr [[TEMP1]], [[TEMP2]]
// CHECK-LABEL: }
