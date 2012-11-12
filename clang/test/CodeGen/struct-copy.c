// RUN: %clang_cc1 -emit-llvm %s -o - | grep 'call.*llvm.memcpy'
struct x { int a[100]; };


void foo(struct x *P, struct x *Q) {
  *P = *Q;
}
