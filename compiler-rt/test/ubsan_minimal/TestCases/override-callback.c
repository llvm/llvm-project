// RUN: %clang -fsanitize=implicit-integer-sign-change %s -o %t && %run %t 0 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CHECK_NO_MESSAGE
// RUN: %clang -fsanitize=implicit-integer-sign-change %s -o %t && %run %t 1 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CHECK_MESSAGE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

static int Result;

int __sanitizer_report_ubsan_error(uintptr_t caller, const char* name) {
  fprintf(stderr, "CUSTOM_CALLBACK\n");
  return Result;
}

void test_message() {
  int32_t t0 = (~((uint32_t)0));
// CHECK_MESSAGE: CUSTOM_CALLBACK
// CHECK_MESSAGE: ubsan: implicit-conversion
}

void test_no_message() {
  int32_t t0 = (~((uint32_t)0));
// CHECK_NO_MESSAGE: CUSTOM_CALLBACK
// CCHECK_NO_MESSAGE-NOT: ubsan: implicit-conversion
}

int main(int argc, const char** argv) {
  Result = atoi(argv[1]);
  if (Result) test_message();
  else test_no_message();
  return 0;
}
