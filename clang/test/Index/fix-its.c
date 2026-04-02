// RUN: c-index-test -test-load-source all -fspell-checking %s 2> %t
// RUN: FileCheck %s < %t
#define MACRO(X) X

int printf(const char *restrict, ...);

void f2() {
  unsigned long index;
  // CHECK: warning: format specifies type 'int' but the argument has type 'unsigned long'
  // CHECK: FIX-IT: Replace [11:17 - 11:19] with "%lu"
  MACRO(printf("%d", index));
}
