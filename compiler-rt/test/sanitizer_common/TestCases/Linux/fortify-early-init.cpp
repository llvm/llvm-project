// Check the fortified _chk interceptors before initialization has finished.
// __<tool>_default_options() is called from the runtime's flag initialization,
// which runs before the interceptors are installed, so REAL() is still null
// there and the interceptor has to take its early startup path instead.

// RUN: %clangxx -O0 %s -o %t && %run %t
// RUN: %clangxx -O2 %s -o %t && %run %t

// REQUIRES: glibc

#include <assert.h>
#include <stddef.h>

extern "C" void *__memset_chk(void *dest, int c, size_t len, size_t destlen);

static char buf[16];

static const char *test() __attribute__((disable_sanitizer_instrumentation)) {
  // Keep the sizes opaque so that the bound check is not folded away.
  volatile size_t len = sizeof(buf) / 2;
  volatile size_t destlen = sizeof(buf);
  __memset_chk(buf, 42, len, destlen);
  return "";
}

extern "C" {
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
const char *__msan_default_options()
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
}

int main(int argc, char *argv[]) {
  // The write happened during initialization and stayed within the bound.
  for (size_t i = 0; i < sizeof(buf) / 2; ++i)
    assert(buf[i] == 42);
  for (size_t i = sizeof(buf) / 2; i < sizeof(buf); ++i)
    assert(buf[i] == 0);
  return 0;
}
