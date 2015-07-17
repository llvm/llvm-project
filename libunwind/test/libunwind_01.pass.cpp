#include <libunwind.h>
#include <stdlib.h>

void backtrace(int lower_bound) {
  unw_context_t context;
  unw_getcontext(&context);

  unw_cursor_t cursor;
  unw_init_local(&cursor, &context);

  int n = 0;
  do {
    ++n;
    if (n > 100) {
      abort();
    }
  } while (unw_step(&cursor) > 0);

  if (n < lower_bound) {
    abort();
  }
}

void test1(int i) {
  backtrace(i);
}

void test2(int i, int j) {
  backtrace(i);
  test1(j);
}

void test3(int i, int j, int k) {
  backtrace(i);
  test2(j, k);
}

int main() {
  test1(1);
  test2(1, 2);
  test3(1, 2, 3);
}
