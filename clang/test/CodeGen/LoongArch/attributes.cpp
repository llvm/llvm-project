// RUN: %clang_cc1 -emit-llvm -triple loongarch64 %s -o - | FileCheck %s

// CHECK: @_ZL2v1 ={{.*}} global i32 0, code_model "small"
static int v1 __attribute__((model("normal")));

void use1() {
  v1 = 1;
}

// CHECK: @v2 ={{.*}} global i32 0, code_model "medium"
int v2 __attribute__((model("medium")));

// CHECK: @v3 ={{.*}} global float 0.000000e+00, code_model "large"
float v3 __attribute__((model("extreme")));

// CHECK: @_ZL2v4IiE ={{.*}} global i32 0, code_model "medium"
template <typename T>
static T v4 __attribute__((model("medium")));

void use2() {
  v4<int> = 1;
}

struct S {
  double d;
};

// CHECK: @v5 ={{.*}} global {{.*}}, code_model "medium"
S v5 __attribute__((model("medium")));

typedef void (*F)();

// CHECK: @v6 ={{.*}} global ptr null, code_model "large"
F v6 __attribute__((model("extreme")));
