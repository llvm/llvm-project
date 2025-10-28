// RUN: %clang_cc1 %s -triple=%itanium_abi_triple -O1 -emit-llvm -fextend-variable-liveness -o - | FileCheck %s
// Make sure we don't crash compiling a lambda that is not nested in a function.
// We also check that fake uses are properly issued in lambdas.

int glob;

extern int foo();

struct S {
  static const int a;
};

const int S::a = [](int b) __attribute__((noinline)) {
  return b * foo();
}
(glob);

int func(int param) {
  return ([=](int lambdaparm) __attribute__((noinline))->int {
    int lambdalocal = lambdaparm * 2;
    return lambdalocal;
  }(glob));
}

// We are looking for the first lambda's call operator, which should contain
// 2 fake uses, one for 'b' and one for its 'this' pointer (in that order).
// The mangled function name contains a $_0, followed by 'cl'.
// This lambda is an orphaned lambda, i.e. one without lexical parent.
//
// CHECK-LABEL: define internal {{.+\"_Z.+\$_0.*cl.*\"}}
// CHECK-NOT:   ret
// CHECK:       fake.use(i32
// CHECK-NOT:   ret
// CHECK:       fake.use(ptr

// The second lambda. We are looking for 3 fake uses.
// CHECK-LABEL: define internal {{.+\"_Z.+\$_0.*cl.*\"}}
// CHECK-NOT:   ret
// CHECK:       fake.use(i32
// CHECK-NOT:   ret
// CHECK:       fake.use(i32
// CHECK-NOT:   ret
// CHECK:       fake.use(ptr
