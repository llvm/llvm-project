// RUN: %clang -O0 %s -o %t && %run %t

// FIXME: TSAN does not use DlsymAlloc.
// UNSUPPORTED: tsan

// FIXME: https://github.com/llvm/llvm-project/pull/106912
// XFAIL: lsan

#include <stdlib.h>

const char *test() __attribute__((disable_sanitizer_instrumentation)) {
  void *volatile p = malloc(3);
  p = realloc(p, 7);
  free(p);

  p = calloc(3, 7);
  free(p);

  free(NULL);

  return "";
}

const char *__asan_default_options()
    __attribute__((disable_sanitizer_instrumentation)) {
  return test();
}
const char *__hwasan_default_options()
    __attribute__((disable_sanitizer_instrumentation)) {
  return test();
}
const char *__lsan_default_options()
    __attribute__((disable_sanitizer_instrumentation)) {
  return test();
}
const char *__memprof_default_options()
    __attribute__((disable_sanitizer_instrumentation)) {
  return test();
}
const char *__msan_default_options()
    __attribute__((disable_sanitizer_instrumentation)) {
  return test();
}
const char *__nsan_default_options()
    __attribute__((disable_sanitizer_instrumentation)) {
  return test();
}
const char *__rtsan_default_options()
    __attribute__((disable_sanitizer_instrumentation)) {
  return test();
}
const char *__tsan_default_options()
    __attribute__((disable_sanitizer_instrumentation)) {
  return test();
}
const char *__ubsan_default_options()
    __attribute__((disable_sanitizer_instrumentation)) {
  return test();
}

int main(int argc, char **argv) { return 0; }
