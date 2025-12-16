// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir

enum Numbers {
  Zero,
  One,
  Two,
  Three
};

int f() {
  return Numbers::One;
}

// CHECK: cir.func{{.*}} @_Z1fv
// CHECK:    cir.const #cir.int<1> : !u32i

namespace test {
  using enum Numbers;
};

int f2() {
  return test::Two;
}

// CHECK: cir.func{{.*}} @_Z2f2v
// CHECK:    cir.const #cir.int<2> : !u32i
