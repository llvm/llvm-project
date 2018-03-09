// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-icall -fsanitize-trap=cfi-icall -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=ITANIUM %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fsanitize=cfi-icall -fsanitize-trap=cfi-icall -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=MS %s

// Tests that we assign unnamed metadata nodes to functions whose types have
// internal linkage.

namespace {

struct S {};

void f(S s) {
}

}

void g() {
  struct S s;
  void (*fp)(S) = f;
  // CHECK: call i1 @llvm.type.test(i8* {{.*}}, metadata [[VOIDS1:![0-9]+]])
  fp(s);
}

// ITANIUM: define internal void @_ZN12_GLOBAL__N_11fENS_1SE({{.*}} !type [[TS1:![0-9]+]] !type [[TS2:![0-9]+]]
// MS: define internal void @"\01?f@?A@@YAXUS@?A@@@Z"({{.*}} !type [[TS1:![0-9]+]] !type [[TS2:![0-9]+]]

// CHECK: [[VOIDS1]] = distinct !{}
// CHECK: [[TS1]] = !{i64 0, [[VOIDS1]]}
// CHECK: [[TS2]] = !{i64 0, [[VOIDS2:![0-9]+]]}
// CHECK: [[VOIDS2]] = distinct !{}
