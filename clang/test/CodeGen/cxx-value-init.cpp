// RUN: %clang_cc1 -emit-llvm %s -o %t

enum E {};
int v1 = E();
float v2 = float();

void f() {
  int v3 = int();
  _Complex int v4 = typeof(_Complex int)();
  _Complex float v5 = typeof(_Complex float)();
}
