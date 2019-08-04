// Tests that __msan_unpoison_param() works as specified.  To prevent MSan
// instrumentation from modifying parameter shadow before each call to foo(), we
// compile main() without MSan.

// RUN: %clangxx_msan -fno-sanitize=memory -c %s -o %t-main.o
// RUN: %clangxx_msan %t-main.o %s -o %t
// RUN: %run %t

#include <assert.h>
#include <sanitizer/msan_interface.h>

#if __has_feature(memory_sanitizer)

__attribute__((noinline)) int bar(int a, int b) {
  volatile int zero = 0;
  return zero;
}

int foo(int a, int b, int unpoisoned_params) {
  if (unpoisoned_params == 0) {
    assert(__msan_test_shadow(&a, sizeof(a)) == 0);
    assert(__msan_test_shadow(&b, sizeof(b)) == 0);
  } else if (unpoisoned_params == 1) {
    assert(__msan_test_shadow(&a, sizeof(a)) == -1);
    assert(__msan_test_shadow(&b, sizeof(b)) == 0);
  } else if (unpoisoned_params == 2) {
    assert(__msan_test_shadow(&a, sizeof(a)) == -1);
    assert(__msan_test_shadow(&b, sizeof(b)) == -1);
  }

  // Poisons parameter shadow in TLS so that the next call from uninstrumented
  // main has params 1 and 2 poisoned no matter what.
  int x, y;
  return bar(x, y);
}

#else

int foo(int, int, int);

int main() {
  foo(0, 0, 2); // Poison parameters for next call.
  foo(0, 0, 0); // Check that both params are poisoned.
  __msan_unpoison_param(1);
  foo(0, 0, 1); // Check that only first param is unpoisoned.
  __msan_unpoison_param(2);
  foo(0, 0, 2); // Check that first and second params are unpoisoned.
  return 0;
}

#endif
