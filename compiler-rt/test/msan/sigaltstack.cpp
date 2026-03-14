// RUN: %clangxx_msan -O0 -g %s -o %t && %run %t
//
#include <signal.h>
#include <assert.h>

#include <sanitizer/msan_interface.h>

int main(void) {
  stack_t old_ss;

  assert(sigaltstack(nullptr, &old_ss) == 0);
  __msan_check_mem_is_initialized(&old_ss, sizeof(stack_t));

  return 0;
}
