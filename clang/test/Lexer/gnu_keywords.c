// RUN: %clang_cc1 -DGNU_KEYWORDS -std=gnu89 -fsyntax-only -verify %s
// RUN: %clang_cc1 -DGNU_KEYWORDS -std=c99 -fgnu-keywords -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c99 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=gnu89 -fno-gnu-keywords -fsyntax-only -verify %s
// expected-no-diagnostics

void f() {
#ifdef GNU_KEYWORDS
  asm ("ret" : :);
#else
  int asm;
#endif
}
