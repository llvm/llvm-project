// RUN: %clangxx_msan -O0 -g %s -o %t && %run %t

#include <assert.h>
#include <sanitizer/msan_interface.h>
#include <semaphore.h>

int main(void) {
  sem_t sem;
  int res = sem_init(&sem, 0, 42);
  assert(res == 0);

  int v;
  res = sem_getvalue(&sem, &v);
  assert(res == 0);
  __msan_check_mem_is_initialized(&v, sizeof(v));
  assert(v == 42);

  res = sem_destroy(&sem);
  assert(res == 0);

  return 0;
}
