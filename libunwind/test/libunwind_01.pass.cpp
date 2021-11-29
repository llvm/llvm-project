// TODO: Investigate these failures on x86_64 macOS back deployment
// UNSUPPORTED: target=x86_64-apple-darwin{{.+}}

#include <libunwind.h>
#include <stdlib.h>
#include <string.h>

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

void test_reg_names() {
  unw_context_t context;
  unw_getcontext(&context);

  unw_cursor_t cursor;
  unw_init_local(&cursor, &context);

  int max_reg_num = -100;
#if defined(__i386__)
  max_reg_num = 7;
#elif defined(__x86_64__)
  max_reg_num = 32;
#endif

  const char prefix[] = "unknown";
  for (int i = -2; i < max_reg_num; ++i) {
    if (strncmp(prefix, unw_regname(&cursor, i), sizeof(prefix) - 1) == 0)
      abort();
  }

  if (strncmp(prefix, unw_regname(&cursor, max_reg_num + 1),
              sizeof(prefix) - 1) != 0)
    abort();
}

#if defined(__x86_64__)
void test_reg_get_set() {
  unw_context_t context;
  unw_getcontext(&context);

  unw_cursor_t cursor;
  unw_init_local(&cursor, &context);

  for (int i = 0; i < 17; ++i) {
    const unw_word_t set_value = 7;
    if (unw_set_reg(&cursor, i, set_value) != UNW_ESUCCESS)
      abort();

    unw_word_t get_value = 0;
    if (unw_get_reg(&cursor, i, &get_value) != UNW_ESUCCESS)
      abort();

    if (set_value != get_value)
      abort();
  }
}

void test_fpreg_get_set() {
  unw_context_t context;
  unw_getcontext(&context);

  unw_cursor_t cursor;
  unw_init_local(&cursor, &context);

  // get/set is not implemented for x86_64 fpregs.
  for (int i = 17; i < 33; ++i) {
    const unw_fpreg_t set_value = 7;
    if (unw_set_fpreg(&cursor, i, set_value) != UNW_EBADREG)
      abort();

    unw_fpreg_t get_value = 0;
    if (unw_get_fpreg(&cursor, i, &get_value) != UNW_EBADREG)
      abort();
  }
}
#else
void test_reg_get_set() {}
void test_fpreg_get_set() {}
#endif

int main(int, char**) {
  test1(1);
  test2(1, 2);
  test3(1, 2, 3);
  test_no_info();
  test_reg_names();
  test_reg_get_set();
  test_fpreg_get_set();
  return 0;
}
