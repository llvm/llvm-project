// RUN: %clangxx_asan %s -pthread -o %t
// RUN: %env_asan_opts=detect_stack_use_after_return=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK1
// RUN: %env_asan_opts=detect_stack_use_after_return=0 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK0
// RUN: %run %t 2>&1 | FileCheck %s --check-prefix=CHECK1
// RUN: %clangxx_asan -O3 %s -pthread -o %t
// RUN: %env_asan_opts=detect_stack_use_after_return=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK1
// RUN: %env_asan_opts=detect_stack_use_after_return=0 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK0
// RUN: %run %t 2>&1 | FileCheck %s --check-prefix=CHECK1
// REQUIRES: stable-runtime

#include <assert.h>
#include <stdio.h>
#include <pthread.h>
#include <sanitizer/asan_interface.h>

static const int kNumThreads = 2;
static const int kLeftRedzoneSize = sizeof(void *) * 4;

void *Thread(void *unused)  {
  void *fake_stack = __asan_get_current_fake_stack();
  char var[15];
  if (fake_stack) {
    fprintf(stderr, "fake stack found: %p; var: %p\n", fake_stack, var);
    // CHECK1: fake stack found
    // CHECK1: fake stack found
    void *beg, *end;
    void *real_stack =
        __asan_addr_is_in_fake_stack(fake_stack, &var[0], &beg, &end);
    assert(real_stack);
    assert((char*)beg <= (char*)&var[0]);
    assert((char*)end > (char*)&var[0]);
    for (int i = -kLeftRedzoneSize; i < 15; i++) {
      void *beg1, *end1;
      char *ptr = &var[0] + i;
      void *real_stack1 =
          __asan_addr_is_in_fake_stack(fake_stack, ptr, &beg1, &end1);
      assert(real_stack == real_stack1);
      assert(beg == beg1);
      assert(end == end1);
    }
  } else {
    fprintf(stderr, "no fake stack\n");
    // CHECK0: no fake stack
    // CHECK0: no fake stack
  }
  return NULL;
}

int main(int argc, char **argv) {
  pthread_t t[kNumThreads];
  for (int i = 0; i < kNumThreads; i++)
    pthread_create(&t[i], 0, Thread, 0);
  for (int i = 0; i < kNumThreads; i++)
    pthread_join(t[i], 0);
  return 0;
}
