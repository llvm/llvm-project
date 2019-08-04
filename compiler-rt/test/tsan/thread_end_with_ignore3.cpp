// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
extern "C" void AnnotateIgnoreReadsBegin(const char *f, int l);
extern "C" void AnnotateIgnoreReadsEnd(const char *f, int l);

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
// CHECK:     #1 main {{.*}}thread_end_with_ignore3.cpp:10
// CHECK:   Ignore was enabled at:
// CHECK:     #0 AnnotateIgnoreReadsBegin
// CHECK:     #1 main {{.*}}thread_end_with_ignore3.cpp:11

