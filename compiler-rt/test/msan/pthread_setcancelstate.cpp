// RUN: %clangxx_msan -O0 %s -o %t && %run %t

#include <assert.h>
#include <pthread.h>
#include <sanitizer/msan_interface.h>

int main(void) {
  int oldstate;
  int oldtype;
  int res = pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &oldstate);
  assert(res == 0);
  __msan_check_mem_is_initialized(&oldstate, sizeof(oldstate));

  res = pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, &oldtype);
  assert(res == 0);
  __msan_check_mem_is_initialized(&oldtype, sizeof(oldtype));

  return 0;
}
