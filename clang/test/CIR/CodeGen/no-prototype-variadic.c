// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=gnu89 -Wno-implicit-function-declaration -fclangir -emit-cir %s -o - | FileCheck %s

int foo(const char *);

int bar(void) {
  int t = foo("x");
  printf("Hello %d\n", t);
  printf("Hi\n");
  return 0;
}

// CHECK: cir.func private @printf(!cir.ptr<!s8i>, ...) -> !s32i