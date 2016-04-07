// Test __sanitizer_coverage_pc_buffer().

// RUN: %clangxx_asan -fsanitize-coverage=edge %s -o %t && %run %t

// UNSUPPORTED: android

#include <assert.h>
#include <sanitizer/coverage_interface.h>
#include <stdio.h>

static volatile int sink;
__attribute__((noinline)) void bar() { sink = 2; }
__attribute__((noinline)) void foo() { sink = 1; }

void assertNotZeroPcs(uintptr_t *buf, uintptr_t size) {
  assert(buf);
  for (uintptr_t i = 0; i < size; ++i)
    assert(buf[i]);
}

int main() {
  uintptr_t *buf = NULL;
  uintptr_t sz = __sanitizer_get_coverage_pc_buffer(&buf);
  assertNotZeroPcs(buf, sz);
  assert(sz);

  foo();
  bar();
  uintptr_t *buf1 = NULL;
  uintptr_t sz1 = __sanitizer_get_coverage_pc_buffer(&buf1);
  assertNotZeroPcs(buf1, sz1);
  assert(buf1 == buf);
  assert(sz1 > sz);

  __sanitizer_reset_coverage();
  uintptr_t *buf3 = NULL;
  uintptr_t sz3 = __sanitizer_get_coverage_pc_buffer(&buf3);
  assertNotZeroPcs(buf3, sz3);
  assert(buf3 == buf);
  assert(sz3 < sz1);
}
