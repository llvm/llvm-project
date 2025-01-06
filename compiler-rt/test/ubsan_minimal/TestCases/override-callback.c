// RUN: %clang -fsanitize=implicit-integer-sign-change %s -o %t && %run %t 0 2>&1 | FileCheck %s
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int Result;

void __ubsan_report_error(const char *kind, uintptr_t caller) {
  fprintf(stderr, "CUSTOM_CALLBACK: %s\n", kind);
}

int main(int argc, const char **argv) {
  int32_t t0 = (~((uint32_t)0));
  // CHECK: CUSTOM_CALLBACK: implicit-conversion
}
