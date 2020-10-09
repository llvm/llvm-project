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

void test_no_info() {
  unw_context_t context;
  unw_getcontext(&context);

  unw_cursor_t cursor;
  unw_init_local(&cursor, &context);

  unw_proc_info_t info;
  int ret = unw_get_proc_info(&cursor, &info);
  if (ret != UNW_ESUCCESS)
    abort();

  // Set the IP to an address clearly outside any function.
  unw_set_reg(&cursor, UNW_REG_IP, (unw_word_t)0);

  ret = unw_get_proc_info(&cursor, &info);
  if (ret != UNW_ENOINFO)
    abort();
}

int main(int, char**) {
  test1(1);
  test2(1, 2);
  test3(1, 2, 3);
  test_no_info();
  return 0;
}
