// RUN: not c-index-test -test-load-source all %s 2> %t.err
// RUN: FileCheck < %t.err -check-prefix=CHECK-LOAD-SOURCE-CRASH %s
// CHECK-LOAD-SOURCE-CRASH: Unable to load translation unit
// RUN: env LIBCLANG_DISABLE_CRASH_RECOVERY=1 not --crash c-index-test -test-load-source all %s
//
// REQUIRES: crash-recovery
// UNSUPPORTED: libstdcxx-safe-mode

#pragma clang __debug crash
