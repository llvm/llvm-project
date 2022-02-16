// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null


void foo() {}

void bar(void) {
  foo(1, 2, 3);  /* Too many arguments passed */
}
