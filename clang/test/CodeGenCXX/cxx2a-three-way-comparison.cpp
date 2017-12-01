// RUN: %clang_cc1 -std=c++2a -emit-llvm %s -o - -triple %itanium_abi_triple | FileCheck %s --check-prefix=ITANIUM
// RUN: not %clang_cc1 -std=c++2a -emit-llvm %s -o - -triple %ms_abi_triple 2>&1 | FileCheck %s --check-prefix=MSABI
// MSABI: cannot mangle this three-way comparison operator yet

struct A {
  void operator<=>(int);
};

// ITANIUM: define {{.*}}@_ZN1AssEi(
void A::operator<=>(int) {}

// ITANIUM: define {{.*}}@_Zssi1A(
void operator<=>(int, A) {}
