// RUN: %clang_cc1 %s -emit-llvm -triple %itanium_abi_triple -o - | FileCheck %s
struct a {
  static void f();
};

void g(a *a) {
  // CHECK: call {{.*}}void @_ZN1a1fEv()
  a->f();
}
