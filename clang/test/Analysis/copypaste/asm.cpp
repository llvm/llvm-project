// RUN: %clang_cc1 -triple x86_64-unknown-linux -analyze -analyzer-checker=alpha.clone.CloneChecker -verify %s

// expected-no-diagnostics

int foo1(int src) {
  int dst = src;
  if (src < 100 && src > 0) {

    asm ("mov %1, %0\n\t"
         "add $1, %0"
         : "=r" (dst)
         : "r" (src));

  }
  return dst;
}

// Identical to foo1 except that it adds two instead of one, so it's no clone.
int foo2(int src) {
  int dst = src;
  if (src < 100 && src > 0) {

    asm ("mov %1, %0\n\t"
         "add $2, %0"
         : "=r" (dst)
         : "r" (src));

  }
  return dst;
}

// Identical to foo1 except that its a volatile asm statement, so it's no clone.
int foo3(int src) {
  int dst = src;
  if (src < 100 && src > 0) {

    asm volatile ("mov %1, %0\n\t"
         "add $1, %0"
         : "=r" (dst)
         : "r" (src));

  }
  return dst;
}
