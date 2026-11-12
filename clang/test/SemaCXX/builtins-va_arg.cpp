// RUN: %clang_cc1 %s -ffreestanding
// RUN: %clang_cc1 %s -ffreestanding -triple i686-unknown-linux
// RUN: %clang_cc1 %s -ffreestanding -triple x86_64-unknown-linux
// RUN: %clang_cc1 %s -ffreestanding -triple mips-unknown-linux
// RUN: %clang_cc1 %s -ffreestanding -triple mipsel-unknown-linux
// RUN: %clang_cc1 %s -ffreestanding -triple armv7-unknown-linux-gnueabi
// RUN: %clang_cc1 %s -ffreestanding -triple thumbv7-unknown-linux-gnueabi

#include "stdarg.h"

int int_accumulator = 0;
double double_accumulator = 0;

int test_vprintf(const char *fmt, va_list ap) {
  char ch;
  int result = 0;
  while (*fmt != '\0') {
    ch = *fmt++;
    if (ch != '%') {
      continue;
    }

    ch = *fmt++;
    switch (ch) {
    case 'd':
      int_accumulator += va_arg(ap, int);
      result++;
      break;

    case 'f':
      double_accumulator += va_arg(ap, double);
      result++;
      break;

    default:
      break;
    }

    if (ch == '0') {
      break;
    }
  }
  return result;
}

int test_printf(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int result = test_vprintf(fmt, ap);
  va_end(ap);
  return result;
}
