// RUN: %clang_min_runtime -fsanitize=implicit-integer-sign-change                                        %s -o %t &&             %run %t 2>&1 | FileCheck %s
// RUN: %clang_min_runtime -fsanitize=implicit-integer-sign-change -fsanitize-handler-preserve-all-regs-experimental            %s -o %t &&             %run %t 2>&1 | FileCheck %s --check-prefixes=PRESERVE
// RUN: %clang_min_runtime -fsanitize=implicit-integer-sign-change -fno-sanitize-recover=all              %s -o %t && not --crash %run %t 2>&1 | FileCheck %s
// RUN: %clang_min_runtime -fsanitize=implicit-integer-sign-change -fno-sanitize-recover=all -DOVERRIDE=1 %s -o %t && not --crash %run %t 2>&1 | FileCheck %s --check-prefixes=FATAL

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int Result;

void __ubsan_report_error(const char *kind, uintptr_t caller) {
  fprintf(stderr, "CUSTOM_CALLBACK: %s\n", kind);
}

[[clang::preserve_all]] void __ubsan_report_error_preserve(const char *kind,
                                                           uintptr_t caller) {
  fprintf(stderr, "CUSTOM_CALLBACK_PRESERVE: %s\n", kind);
}

#if OVERRIDE
void __ubsan_report_error_fatal(const char *kind, uintptr_t caller) {
  fprintf(stderr, "FATAL_CALLBACK: %s\n", kind);
}
#endif

int main(int argc, const char **argv) {
  int32_t t0 = (~((uint32_t)0));
  // CHECK: CUSTOM_CALLBACK: implicit-conversion
  // PRESERVE: CUSTOM_CALLBACK_PRESERVE: implicit-conversion
  // FATAL: FATAL_CALLBACK: implicit-conversion
}
