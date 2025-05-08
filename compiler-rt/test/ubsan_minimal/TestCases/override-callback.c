// RUN: %clang -fsanitize=implicit-integer-sign-change                           %s -o %t &&             %run %t 2>&1 | FileCheck %s
// RUN: %clang -fsanitize=implicit-integer-sign-change -fno-sanitize-recover=all %s -o %t && not --crash %run %t 2>&1 | FileCheck %s --check-prefixes=FATAL

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int Result;

void __ubsan_report_error(const char *kind, uintptr_t caller) {
  fprintf(stderr, "CUSTOM_CALLBACK: %s\n", kind);
}

void __ubsan_report_error_fatal(const char *kind, uintptr_t caller) {
  fprintf(stderr, "FATAL_CALLBACK: %s\n", kind);
}

int main(int argc, const char **argv) {
  int32_t t0 = (~((uint32_t)0));
  // CHECK: CUSTOM_CALLBACK: implicit-conversion
  // FATAL: FATAL_CALLBACK: implicit-conversion
}
