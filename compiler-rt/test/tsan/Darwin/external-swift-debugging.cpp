// RUN: %clangxx_tsan %s -o %t
// RUN: %deflake %run %t 2>&1 | FileCheck %s

#include <dlfcn.h>
#include <thread>

#import "../test.h"


extern "C" {
int __tsan_get_report_data(void *report, const char **description, int *count,
                           int *stack_count, int *mop_count, int *loc_count,
                           int *mutex_count, int *thread_count,
                           int *unique_tid_count, void **sleep_trace,
                           unsigned long trace_size);
int __tsan_get_report_tag(void *report, unsigned long *tag);
int __tsan_get_report_mop(void *report, unsigned long idx, int *tid, void **addr,
                          int *size, int *write, int *atomic, void **trace,
                          unsigned long trace_size);
}

__attribute__((no_sanitize("thread"), noinline))
void ExternalWrite(void *addr) {
  void *kSwiftAccessRaceTag = (void *)0x1;
  __tsan_external_write(addr, nullptr, kSwiftAccessRaceTag);
}

int main(int argc, char *argv[]) {
  barrier_init(&barrier, 2);
  fprintf(stderr, "Start.\n");
  // CHECK: Start.

  void *opaque_object = malloc(16);
  std::thread t1([opaque_object] {
    ExternalWrite(opaque_object);
    barrier_wait(&barrier);
  });
  std::thread t2([opaque_object] {
    barrier_wait(&barrier);
    ExternalWrite(opaque_object);
  });
  // CHECK: WARNING: ThreadSanitizer: Swift access race
  // CHECK: Modifying access of Swift variable at {{.*}} by thread {{.*}}
  // CHECK:   #0 ExternalWrite
  // CHECK: Previous modifying access of Swift variable at {{.*}} by thread {{.*}}
  // CHECK:   #0 ExternalWrite
  // CHECK: SUMMARY: ThreadSanitizer: Swift access race
  t1.join();
  t2.join();

  fprintf(stderr, "Done.\n");
}

extern "C" __attribute__((disable_sanitizer_instrumentation)) void
__tsan_on_report(void *report) {
  const char *description;
  int count;
  int stack_count, mop_count, loc_count, mutex_count, thread_count,
      unique_tid_count;
  void *sleep_trace[16] = {0};
  __tsan_get_report_data(report, &description, &count, &stack_count, &mop_count,
                         &loc_count, &mutex_count, &thread_count,
                         &unique_tid_count, sleep_trace, 16);
  fprintf(stderr, "report type = '%s', count = %d, mop_count = %d\n", description, count, mop_count);
  // CHECK: report type = 'external-race', count = 0, mop_count = 2

  unsigned long tag;
  __tsan_get_report_tag(report, &tag);
  fprintf(stderr, "tag = %ld\n", tag);
  // CHECK: tag = 1

  int tid, size, write, atomic;
  void *addr;
  void *trace[16] = {0};
  __tsan_get_report_mop(report, /*idx=*/0, &tid, &addr, &size, &write, &atomic,
                        trace, 16);
  fprintf(stderr, "Racy write trace (1 of 2):\n");
  for (int i = 0; i < 16 && trace[i]; i++) {
    Dl_info info;
    dladdr(trace[i], &info);
    fprintf(stderr, "  %d: frame: %p, function: %p %s\n", i, trace[i],
            info.dli_saddr, info.dli_sname);
  }
  // Ensure ExternalWrite() function is top of trace
  // CHECK: 0: frame: 0x{{[0-9a-z]+}}, function: 0x{{[0-9a-z]+}} _Z13ExternalWritePv
}

// CHECK: Done.
// CHECK: ThreadSanitizer: reported 1 warnings
