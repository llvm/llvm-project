// RUN: %clang_cc1 -emit-llvm %s -o - -triple=powerpc-unknown-linux | FileCheck %s

struct S {
  S();
  ~S();
};

void byval(S one, S two) {
  one = two;
}

// CHECK: define void @_Z5byval1SS_(%struct.S* %one, %struct.S* %two)
