// RUN: %clang_cc1 -std=c++2a -emit-llvm %s -o - -triple %itanium_abi_triple | FileCheck %s --check-prefixes=CHECK,ITANIUM
// RUN: %clang_cc1 -std=c++2a -emit-llvm %s -o - -triple %ms_abi_triple | FileCheck %s --check-prefixes=CHECK,MSABI
// RUN: not %clang_cc1 -std=c++2a -emit-llvm %s -o - -triple %itanium_abi_triple -DBUILTIN 2>&1 | FileCheck %s --check-prefix=BUILTIN
// RUN: not %clang_cc1 -std=c++2a -emit-llvm %s -o - -triple %ms_abi_triple -DBUILTIN 2>&1 | FileCheck %s --check-prefix=BUILTIN

struct A {
  void operator<=>(int);
};

// ITANIUM: define {{.*}}@_ZN1AssEi(
// MSABI: define {{.*}}@"??__MA@@QEAAXH@Z"(
void A::operator<=>(int) {}

// ITANIUM: define {{.*}}@_Zssi1A(
// MSABI: define {{.*}}@"??__M@YAXHUA@@@Z"(
void operator<=>(int, A) {}

int operator<=>(A, A);

// ITANIUM: define {{.*}}_Z1f1A(
// MSABI: define {{.*}}"?f@@YAHUA@@@Z"(
int f(A a) {
  // ITANIUM: %[[RET:.*]] = call {{.*}}_Zss1AS_(
  // MSABI: %[[RET:.*]] = call {{.*}}"??__M@YAHUA@@0@Z"(
  // CHECK: ret i32 %[[RET]]
  return a <=> a;
}

#ifdef BUILTIN
void builtin(int a) {
  a <=> a; // BUILTIN: cannot compile this scalar expression yet
}
#endif
