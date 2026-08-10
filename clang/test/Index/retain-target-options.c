// RUN: c-index-test -test-load-source all -target x86_64-apple-darwin10.0.0 -msse4.1 %s 2>&1 | FileCheck %s
// RUN: c-index-test -test-load-source-reparse 1 all -target x86_64-apple-darwin10.0.0 -msse4.1 %s 2>&1 | FileCheck %s
// RUN: c-index-test -test-load-source-reparse 5 all -target x86_64-apple-darwin10.0.0 -msse4.1 %s 2>&1 | FileCheck %s

// CHECK: error: SSE4_1 used
#if defined(__SSE4_1__)
#error SSE4_1 used
#endif
