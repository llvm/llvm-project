// RUN: %clang_cc1 -std=c++98 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++11 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++14 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++17 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++20 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++23 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++2c %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK

namespace cwg519 { // cwg519: 2.7
void f() {
  int *a = 0;
  void *v = a;
  bool c1 = v == static_cast<void *>(0);

  void *w = 0;
  int *b = static_cast<int*>(w);
  bool c2 = b == static_cast<int *>(0);
}
} // namespace cwg519

// We're checking that `null`s that were initially stored in `a` and `w`
// are simply copied over all the way to respective comparisons with `null`.

// CHECK-LABEL: define {{.*}} void @cwg519::f()()
// CHECK:         store ptr null, ptr [[A:%.+]],
// CHECK-NEXT:    [[TEMP_A:%.+]] = load ptr, ptr [[A]] 
// CHECK-NEXT:    store ptr [[TEMP_A]], ptr [[V:%.+]],
// CHECK-NEXT:    [[TEMP_V:%.+]] = load ptr, ptr [[V]]
// CHECK-NEXT:    {{.+}} = icmp eq ptr [[TEMP_V]], null

// CHECK:         store ptr null, ptr [[W:%.+]],
// CHECK-NEXT:    [[TEMP_W:%.+]] = load ptr, ptr [[W]] 
// CHECK-NEXT:    store ptr [[TEMP_W]], ptr [[B:%.+]],
// CHECK-NEXT:    [[TEMP_B:%.+]] = load ptr, ptr [[B]]
// CHECK-NEXT:    {{.+}} = icmp eq ptr [[TEMP_B]], null
// CHECK-LABEL: }
