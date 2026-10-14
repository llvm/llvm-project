// RUN: %clang_cc1 -emit-llvm %s -triple x86_64-unknown-linux-gnu -o - | FileCheck %s

// After a musttail call, the function epilog should not emit a redundant
// return statement in a dead block.

int F1(short);
void V1(int);
double _Complex C1(short);

// CHECK-LABEL: define {{.*}} @_Z5test1s(
// CHECK: musttail call
// CHECK-NEXT: ret i32
// CHECK-NOT: ret i32
int test1(short P0) {
  [[clang::musttail]] return F1(P0);
}

// CHECK-LABEL: define {{.*}} @_Z5test2i(
// CHECK: musttail call
// CHECK-NEXT: ret void
// CHECK-NOT: ret void
void test2(int x) {
  [[clang::musttail]] return V1(x);
}

// CHECK-LABEL: define {{.*}} @_Z5test3s(
// CHECK: musttail call
// CHECK-NEXT: ret { double, double }
// CHECK-NOT: ret { double, double }
double _Complex test3(short P0) {
  [[clang::musttail]] return C1(P0);
}
