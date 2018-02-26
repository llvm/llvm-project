// RUN: %clang_cc1 -std=c++2a -emit-llvm %s -o - -triple %itanium_abi_triple | FileCheck %s --check-prefix=ITANIUM
// RUN: not %clang_cc1 -std=c++2a -emit-llvm %s -o - -triple %ms_abi_triple 2>&1 | FileCheck %s --check-prefix=MSABI
// RUN: not %clang_cc1 -std=c++2a -emit-llvm %s -o - -triple %itanium_abi_triple -DBUILTIN 2>&1 | FileCheck %s --check-prefix=BUILTIN
// MSABI: cannot mangle this three-way comparison operator yet

struct A {
  void operator<=>(int);
};

// ITANIUM: define {{.*}}@_ZN1AssEi(
void A::operator<=>(int) {}

// ITANIUM: define {{.*}}@_Zssi1A(
void operator<=>(int, A) {}

int operator<=>(A, A);

// ITANIUM: define {{.*}}_Z1f1A(
int f(A a) {
  // ITANIUM: %[[RET:.*]] = call {{.*}}_Zss1AS_(
  // ITANIUM: ret i32 %[[RET]]
  return a <=> a;
}

#ifdef BUILTIN
void builtin(int a) {
  a <=> a; // BUILTIN: cannot compile this scalar expression yet
}
#endif
