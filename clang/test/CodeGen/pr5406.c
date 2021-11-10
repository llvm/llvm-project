// REQUIRES: arm-registered-target
// RUN: %clang_cc1 %s -emit-llvm -triple arm-apple-darwin -o - | FileCheck %s
// PR 5406

typedef struct { char x[3]; } A0;
void foo (int i, ...);


// CHECK: call void (i32, ...) @foo(i32 1, [1 x i32] {{.*}})
int main (void)
{
  A0 a3;
  a3.x[0] = 0;
  a3.x[0] = 0;
  a3.x[2] = 26;
  foo (1,  a3 );
  return 0;
}
