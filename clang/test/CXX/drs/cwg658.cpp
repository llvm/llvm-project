// RUN: %clang_cc1 -std=c++98 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++11 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++14 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++17 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++20 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++23 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++2c %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK

namespace cwg658 { // cwg658: 2.7

void f(int* p1) {
  char* p2 = reinterpret_cast<char*>(p1);
}

} // namespace cwg658

// We're checking that p1 is stored into p2 without changes.

// CHECK-LABEL: define {{.*}} void @cwg658::f(int*)(ptr noundef %p1)
// CHECK:         [[P1_ADDR:%.+]] = alloca ptr, align 8
// CHECK-NEXT:    [[P2:%.+]] = alloca ptr, align 8
// CHECK:         store ptr %p1, ptr [[P1_ADDR]]
// CHECK-NEXT:    [[TEMP:%.+]] = load ptr, ptr [[P1_ADDR]]
// CHECK-NEXT:    store ptr [[TEMP]], ptr [[P2]]
// CHECK-LABEL: }
