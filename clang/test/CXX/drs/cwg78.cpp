// RUN: %clang_cc1 -std=c++98 %s -verify -DVERIFY -fexceptions -fcxx-exceptions -pedantic-errors -triple %itanium_abi_triple
// RUN: %clang_cc1 -std=c++11 %s -verify -DVERIFY -fexceptions -fcxx-exceptions -pedantic-errors -triple %itanium_abi_triple
// RUN: %clang_cc1 -std=c++14 %s -verify -DVERIFY -fexceptions -fcxx-exceptions -pedantic-errors -triple %itanium_abi_triple
// RUN: %clang_cc1 -std=c++17 %s -verify -DVERIFY -fexceptions -fcxx-exceptions -pedantic-errors -triple %itanium_abi_triple
// RUN: %clang_cc1 -std=c++20 %s -verify -DVERIFY -fexceptions -fcxx-exceptions -pedantic-errors -triple %itanium_abi_triple
// RUN: %clang_cc1 -std=c++23 %s -verify -DVERIFY -fexceptions -fcxx-exceptions -pedantic-errors -triple %itanium_abi_triple
// RUN: %clang_cc1 -std=c++23 %s -verify -DVERIFY -fexceptions -fcxx-exceptions -pedantic-errors -triple %itanium_abi_triple

// RUN: %clang_cc1 -std=c++98 %s -triple %itanium_abi_triple -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++11 %s -triple %itanium_abi_triple -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++14 %s -triple %itanium_abi_triple -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++17 %s -triple %itanium_abi_triple -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++20 %s -triple %itanium_abi_triple -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++23 %s -triple %itanium_abi_triple -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++2c %s -triple %itanium_abi_triple -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK

namespace cwg78 { // cwg78: partial
// Under CWG78, this is valid, because 'k' has static storage duration, so is
// zero-initialized.
#ifdef VERIFY
const int k;
// expected-error@-1 {{default initialization of an object of const type 'const int'}}
#endif

int f() {
  static int i;
  return i;
}

// Checking that static variable is zero-initialized,
// and not default-initialized (which means no initialization in this case).

// CHECK: @cwg78::f()::i = {{.*}} i32 0
} // namespace cwg78
