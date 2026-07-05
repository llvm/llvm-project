// Test that dispatch continuation memory region is scanned.
// RUN: %clangxx_lsan %s  -o %t -framework Foundation
// RUN: %env_lsan_opts="report_objects=1" %run %t 2>&1 && echo "" | FileCheck %s

#include <dispatch/dispatch.h>
#include <sanitizer/lsan_interface.h>

int main() {
  // Reduced from `CFRunLoopCreate`
  dispatch_queue_t fake_rl_queue = dispatch_get_global_queue(2, 0);
  dispatch_source_t timer =
      dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, fake_rl_queue);
  dispatch_source_set_event_handler(timer, ^{
                                    });
  dispatch_source_set_timer(timer, DISPATCH_TIME_FOREVER, DISPATCH_TIME_FOREVER,
                            321);
  dispatch_resume(timer);
  __lsan_do_leak_check();
  dispatch_source_cancel(timer);
  dispatch_release(timer);
  return 0;
}

// CHECK-NOT: LeakSanitizer: detected memory leaks
