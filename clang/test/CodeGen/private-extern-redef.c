// RUN: %clang_cc1 -triple x86_64-darwin-apple -emit-llvm -o - %s | FileCheck %s
// rdar://9609649

__private_extern__ const int I;
__private_extern__ const int J = 927;

__private_extern__ const int K;
const int K = 37;

const int L = 10;
__private_extern__ const int L;

__private_extern__ int M;
int M = 20;

__private_extern__ int N;
int N;

__private_extern__ int O;
int O=1;

__private_extern__ int P;
extern int P;

void bar(int);

void foo(void) {
  bar(I);
}

// CHECK: @J = hidden constant
// CHECK: @K = hidden constant
// CHECK: @L ={{.*}} constant
// CHECK: @M = hidden global
// CHECK: @O = hidden global
// CHECK: @I = external hidden
// CHECK: @N = hidden global
// CHECK-NOT: @P

