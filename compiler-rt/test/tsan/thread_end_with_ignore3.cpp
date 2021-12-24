// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s

// ReportIgnoresEnabled is disabled on Darwin, see comment in tsan_rtl_thread.cpp.
// UNSUPPORTED: darwin
#include "test.h"

int main() {
  AnnotateIgnoreReadsBegin("", 0);
  AnnotateIgnoreReadsBegin("", 0);
  AnnotateIgnoreReadsEnd("", 0);
  AnnotateIgnoreReadsEnd("", 0);
  AnnotateIgnoreReadsBegin("", 0);
  AnnotateIgnoreReadsBegin("", 0);
  AnnotateIgnoreReadsEnd("", 0);
}

// CHECK: ThreadSanitizer: main thread finished with ignores enabled
// CHECK:   Ignore was enabled at:
// CHECK:     #0 AnnotateIgnoreReadsBegin
// CHECK:     #1 main {{.*}}thread_end_with_ignore3.cpp:12
// CHECK:   Ignore was enabled at:
// CHECK:     #0 AnnotateIgnoreReadsBegin
// CHECK:     #1 main {{.*}}thread_end_with_ignore3.cpp:13
