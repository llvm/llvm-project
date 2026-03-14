// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s

#include "test.h"

extern "C" {
void __tsan_on_report(void *report);
void *__tsan_get_current_report();
int __tsan_get_report_data(void *report, const char **description, int *count,
                           int *stack_count, int *mop_count, int *loc_count,
                           int *mutex_count, int *thread_count,
                           int *unique_tid_count, void **sleep_trace,
                           unsigned long trace_size);
int __tsan_get_report_stack(void *report, unsigned long idx, void **trace,
                            unsigned long trace_size);
int __tsan_get_report_mutex(void *report, unsigned long idx, uint64_t *mutex_id,
                            void **addr, int *destroyed, void **trace,
                            unsigned long trace_size);
}

int main() {
  int m = 0;
  fprintf(stderr, "&m = %p\n", &m);
  // CHECK: &m = [[MUTEX:0x[0-9a-f]+]]
  AnnotateRWLockReleased(__FILE__, __LINE__, &m, 1);
  fprintf(stderr, "Done.\n");
  return 0;
}

// Required for dyld macOS 12.0+
#if (__APPLE__)
__attribute__((weak))
#endif
__attribute__((disable_sanitizer_instrumentation)) extern "C" void
__tsan_on_report(void *report) {
  fprintf(stderr, "__tsan_on_report(%p)\n", report);
  fprintf(stderr, "__tsan_get_current_report() = %p\n",
          __tsan_get_current_report());
  // CHECK: __tsan_on_report([[REPORT:0x[0-9a-f]+]])
  // CHECK: __tsan_get_current_report() = [[REPORT]]

  const char *description;
  int count;
  int stack_count, mop_count, loc_count, mutex_count, thread_count,
      unique_tid_count;
  void *sleep_trace[16] = {0};
  __tsan_get_report_data(report, &description, &count, &stack_count, &mop_count,
                         &loc_count, &mutex_count, &thread_count,
                         &unique_tid_count, sleep_trace, 16);

  fprintf(stderr, "stack_count = %d\n", stack_count);
  // CHECK: stack_count = 1

  fprintf(stderr, "mutex_count = %d\n", mutex_count);
  // CHECK: mutex_count = 1

  void *trace[16] = {0};
  __tsan_get_report_stack(report, 0, trace, 16);

  fprintf(stderr, "trace[0] = %p, trace[1] = %p, trace[2] = %p\n", trace[0],
          trace[1], trace[2]);
  // CHECK: trace[0] = 0x{{[0-9a-f]+}}, trace[1] = 0x{{[0-9a-f]+}}, trace[2] =
  // {{0x0|\(nil\)|\(null\)}}

  uint64_t mutex_id;
  void *addr;
  int destroyed;
  __tsan_get_report_mutex(report, 0, &mutex_id, &addr, &destroyed, trace, 16);
  fprintf(stderr, "addr = %p, destroyed = %d\n", addr, destroyed);
  // CHECK: addr = [[MUTEX]], destroyed = 0
  fprintf(stderr, "trace[0] = %p, trace[1] = %p, trace[2] = %p\n", trace[0],
          trace[1], trace[2]);
  // CHECK: trace[0] = 0x{{[0-9a-f]+}}, trace[1] = 0x{{[0-9a-f]+}}, trace[2] =
  // {{0x0|\(nil\)|\(null\)}}
}

// CHECK: Done.
// CHECK: ThreadSanitizer: reported 1 warnings
