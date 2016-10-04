// Test __sanitizer_coverage_pc_buffer().

// RUN: %clangxx_asan -fsanitize-coverage=edge %stdcxx11 %s -O3 -o %t && %run %t

// UNSUPPORTED: android

#include <assert.h>
#include <memory>
#include <sanitizer/coverage_interface.h>
#include <stdint.h>
#include <stdio.h>

static volatile int sink;
__attribute__((noinline)) void foo() { sink = 1; }

void assertNotZeroPcs(uintptr_t *buf, uintptr_t size) {
  assert(buf);
  for (uintptr_t i = 0; i < size; ++i)
    assert(buf[i]);
}

int main() {
  uintptr_t buf_size = 1 << 20;
  std::unique_ptr<uintptr_t[]> buf(new uintptr_t[buf_size]);
  __sanitizer_set_coverage_pc_buffer(buf.get(), buf_size);

  {
    uintptr_t sz = __sanitizer_get_coverage_pc_buffer_pos();
    assertNotZeroPcs(buf.get(), sz);
    assert(sz);
  }

  {
    uintptr_t sz = __sanitizer_get_coverage_pc_buffer_pos();
    foo();
    uintptr_t sz1 = __sanitizer_get_coverage_pc_buffer_pos();
    assertNotZeroPcs(buf.get(), sz1);
    assert(sz1 > sz);
  }

  {
    uintptr_t sz = __sanitizer_get_coverage_pc_buffer_pos();
    // reset coverage to 0.
    __sanitizer_reset_coverage();
    uintptr_t sz1 = __sanitizer_get_coverage_pc_buffer_pos();
    assertNotZeroPcs(buf.get(), sz1);
    assert(sz1 < sz);
  }
}
