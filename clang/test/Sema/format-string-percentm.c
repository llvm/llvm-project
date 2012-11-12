// RUN: %clang_cc1 -fsyntax-only -verify %s -triple i686-pc-linux-gnu
// expected-no-diagnostics

// PR 4142 - support glibc extension to printf: '%m' (which prints strerror(errno)).
int printf(char const*,...);
void percentm(void) {
  printf("%m");
}
