// RUN: %clang -g -fsanitize=function %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --check-prefix=CHECK --implicit-check-not='runtime error:'

void f(void (*fp)(int (*)[])) { fp(0); }

void callee0(int (*a)[]) {}
void callee1(int (*a)[1]) {}

int main() {
  int a[1];
  f(callee0);
  // CHECK: runtime error: call to function callee1 through pointer to incorrect function type 'void (*)(int (*)[])'
  f(callee1); // compatible type in C, but flagged
}
