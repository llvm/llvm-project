// The same early startup path as fortify-early-init.cpp, but with a size which
// exceeds the destination. REAL() cannot be used to report this yet, so the
// fortification bound is enforced with CHECK_LE instead.

// RUN: %clangxx -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s

// lsan and ubsan do not intercept these, so they just get glibc's own abort,
// which fortify-overflow.cpp already covers. tsan does reach the CHECK, but
// then faults while unwinding this early in startup, so its exit status is not
// stable enough to match on.
// REQUIRES: glibc && (asan || hwasan || msan)

#include <stddef.h>

extern "C" void *__memset_chk(void *dest, int c, size_t len, size_t destlen);

static char buf[16];

static const char *test() __attribute__((disable_sanitizer_instrumentation)) {
  // Keep the sizes opaque so that the bound check is not folded away.
  volatile size_t len = sizeof(buf);
  volatile size_t destlen = sizeof(buf) / 2;
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

// The reported values confirm the destination size was the bound applied. Which
// file reports it differs, as msan supplies its own _chk implementations.
// CHECK: CHECK failed: {{.*}} (0x10, 0x8)

int main(int argc, char *argv[]) {
  // CHECK-NOT: unreachable
  __builtin_printf("unreachable\n");
  return 0;
}
