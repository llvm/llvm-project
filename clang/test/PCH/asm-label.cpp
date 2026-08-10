// RUN: %clang_cc1 -emit-pch %s -o %t
// RUN: %clang_cc1 -include-pch %t %s -verify
#ifndef HEADER_H
#define HEADER_H
template<int = 0> 
void MyMethod() {
  void *bar;
  some_path:
  asm goto
      (
          "mov %w[foo], %w[foo]"
          : [foo] "=r"(bar)
          : [foo2] "r"(bar), [foo3] "r"(bar), [foo4] "r"(bar)
          : 
          : some_path
      );
  }
#else
void test() {
 MyMethod();
// expected-no-diagnostics
}
#endif
