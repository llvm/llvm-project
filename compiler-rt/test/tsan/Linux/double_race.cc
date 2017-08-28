// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t 2>&1 | FileCheck %s
#include "../test.h"
#include <memory.h>

// A reproducer for a known issue.
// See reference to double_race.cc in tsan_rtl_report.cc for an explanation.

char buf[16];
volatile int nreport;

void __sanitizer_report_error_summary(const char *summary) {
  nreport++;
}

const int kEventPCBits = 61;

extern "C" bool __tsan_symbolize_external(unsigned long pc, char *func_buf,
                                          unsigned long func_siz,
                                          char *file_buf,
                                          unsigned long file_siz, int *line,
                                          int *col) {
  if (pc >> kEventPCBits) {
    printf("bad PC passed to __tsan_symbolize_external: %lx\n", pc);
    _exit(1);
  }
  return true;
}

void *Thread(void *arg) {
  barrier_wait(&barrier);
  memset(buf, 2, sizeof(buf));
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  memset(buf, 1, sizeof(buf));
  barrier_wait(&barrier);
  pthread_join(t, 0);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Write of size 8 at {{.*}} by thread T1:
// CHECK:     #0 memset
// CHECK:     #1 Thread
// CHECK-NOT: bad PC passed to __tsan_symbolize_external
// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Write of size 8 at {{.*}} by thread T1:
// CHECK:     #0 Thread
