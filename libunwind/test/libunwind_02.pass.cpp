// TODO: Investigate these failures
// XFAIL: asan, tsan, ubsan

#include <assert.h>
#include <stdlib.h>
#include <unwind.h>

#define EXPECTED_NUM_FRAMES 50
#define NUM_FRAMES_UPPER_BOUND 100

_Unwind_Reason_Code callback(_Unwind_Context *context, void *cnt) {
  (void)context;
  int *i = (int *)cnt;
  ++*i;
  if (*i > NUM_FRAMES_UPPER_BOUND) {
    abort();
  }
  return _URC_NO_REASON;
}

void test_backtrace() {
  int n = 0;
  _Unwind_Backtrace(&callback, &n);
  if (n < EXPECTED_NUM_FRAMES) {
    abort();
  }
}

int test(int i) {
  if (i == 0) {
    test_backtrace();
    return 0;
  } else {
    return i + test(i - 1);
  }
}

int main(int, char**) {
  int total = test(50);
  assert(total == 1275);
  return 0;
}
