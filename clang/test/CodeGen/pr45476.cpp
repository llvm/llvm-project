// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s
// PR45476

// This test used to get into an infinite loop,
// which, in turn, caused clang to never finish execution.

struct s3 {
  char a, b, c;
};

_Atomic struct s3 a;

extern "C" void foo() {
  // CHECK-LABEL: @foo
  // CHECK: store atomic i32

  a = s3{1, 2, 3};
}

