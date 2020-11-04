// Test that executable with ELF-TLS will link/run successfully on aarch64
// RUN: %clangxx -fno-emulated-tls %s -o %t
// RUN: %run %t 2>&1
// REQUIRES: android-28

#include <stdio.h>
#include <stdlib.h>

__thread void *tls_var;
int var;

void set_var() {
  var = 123;
  tls_var = &var;
}
int main() {
  set_var();
  fprintf(stderr, "Test alloc: %p\n", tls_var);
  fflush(stderr);
  return 0;
}
