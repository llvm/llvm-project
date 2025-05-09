// RUN: %clang_cc1 -emit-llvm -triple x86_64-unknown-unknown %s -o - | FileCheck %s

// CHECK: @_ZL2v1 ={{.*}} global i32 0, code_model "small"
static int v1 __attribute__((model("small")));

void use1() {
  v1 = 1;
}

// CHECK: @v2 ={{.*}} global float 0.000000e+00, code_model "large"
float v2 __attribute__((model("large")));

// CHECK: @_ZL2v3IiE ={{.*}} global i32 0, code_model "small"
template <typename T>
static T v3 __attribute__((model("small")));

void use2() {
  v3<int> = 1;
}
struct S {
  double d;
};

typedef void (*F)();

// CHECK: @v4 ={{.*}} global ptr null, code_model "large"
F v4 __attribute__((model("large")));
