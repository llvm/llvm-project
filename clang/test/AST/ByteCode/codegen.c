// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -o - %s -fexperimental-new-constant-interpreter | FileCheck %s

typedef __INTPTR_TYPE__ intptr_t;

const intptr_t Z1 = (intptr_t)(((char*)-1LL) + 1);
// CHECK: @Z1 = constant i64 0

const intptr_t Z2 = (intptr_t)(((char*)1LL) - 1);
// CHECK: @Z2 = constant i64 0

struct A {
  char num_fields;
};
struct B {
  char a, b[1];
};
const int A = (char *)(&( (struct B *)(16) )->b[0]) - (char *)(16);
// CHECK: @A = constant i32 1

struct X { int a[2]; };
int test(void) {
  static int i23 = (int) &(((struct X *)0)->a[1]);
  return i23;
}
// CHECK: @test.i23 = internal global i32 4, align 4
