// RUN: not c-index-test -test-load-source all %s 2> %t.err
// RUN: FileCheck < %t.err -check-prefix=CHECK-LOAD-SOURCE-CRASH %s
// CHECK-LOAD-SOURCE-CRASH: Unable to load translation unit
//
// REQUIRES: crash-recovery

#pragma clang __debug crash
