// RUN: %clang_cc1 -O2 -emit-llvm %s -o /dev/null
void f(_Complex float z);
void g(void) { f(1.0i); }
