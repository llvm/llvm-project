// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-icall -fsanitize-trap=cfi-icall -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=ITANIUM %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fsanitize=cfi-icall -fsanitize-trap=cfi-icall -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=MS %s

// Tests that we assign unnamed metadata nodes to functions whose types have
// internal linkage.

namespace {

struct S {};

void f(S *s) {
}

}

void g() {
  void (*fp)(S *) = f;
  // CHECK: call i1 @llvm.type.test(i8* {{.*}}, metadata [[VOIDS:![0-9]+]])
  fp(0);
}

// ITANIUM: define internal void @_ZN12_GLOBAL__N_11fEPNS_1SE({{.*}} !type [[TS:![0-9]+]]
// MS: define internal void @"\01?f@?A@@YAXPEAUS@?A@@@Z"({{.*}} !type [[TS:![0-9]+]]

// CHECK: [[VOIDS]] = distinct !{}
// CHECK: [[TS]] = !{i64 0, [[VOIDS]]}
