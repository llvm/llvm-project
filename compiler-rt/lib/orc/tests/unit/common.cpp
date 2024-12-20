#include <stdio.h>

/// Defined so that tests can use code that logs errors.
extern "C" void __orc_rt_log_error(const char *ErrMsg) {
  fprintf(stderr, "orc runtime error: %s\n", ErrMsg);
}
