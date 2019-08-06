#pragma once

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define UNTAG(x) (typeof((x) + 0))(((uintptr_t)(x)) & 0xffffffffffffff)

__attribute__((no_sanitize("hwaddress")))
int untag_printf(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int ret = vprintf(UNTAG(fmt), ap);
  va_end(ap);
  return ret;
}

__attribute__((no_sanitize("hwaddress")))
int untag_fprintf(FILE *stream, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int ret = vfprintf(stream, UNTAG(fmt), ap);
  va_end(ap);
  return ret;
}

int untag_strcmp(const char *s1, const char *s2) {
  return strcmp(UNTAG(s1), UNTAG(s2));
}
