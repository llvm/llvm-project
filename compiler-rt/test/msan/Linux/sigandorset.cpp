// RUN: %clangxx_msan -std=c++11 -O0 -g %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_msan -DLEFT_OK -std=c++11 -O0 -g %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_msan -DRIGHT_OK -std=c++11 -O0 -g %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_msan -DLEFT_OK -DRIGHT_OK -std=c++11 -O0 -g %s -o %t && %run %t
// REQUIRES: !android

#include <assert.h>
#include <sanitizer/msan_interface.h>
#include <signal.h>
#include <sys/time.h>
#include <unistd.h>

int main(void) {
  sigset_t s, t, u;
#ifdef LEFT_OK
  sigemptyset(&t);
#endif
#ifdef RIGHT_OK
  sigemptyset(&u);
#endif

  // CHECK:  MemorySanitizer: use-of-uninitialized-value
  // CHECK-NEXT: in main {{.*}}sigandorset.cpp:[[@LINE+1]]
  sigandset(&s, &t, &u);
  sigorset(&s, &t, &u);
  __msan_check_mem_is_initialized(&s, sizeof s);
  return 0;
}
