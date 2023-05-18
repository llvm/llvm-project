// RUN: export LSAN_OPTIONS=detect_leaks=0
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_PREAMBLE_FILE=%t-preamble.pch \
// RUN:   not c-index-test -code-completion-at=%s:20:1 \
// RUN:   "-remap-file=%s,%S/Inputs/crash-recovery-code-complete-remap.c" \
// RUN:   %s 2> %t.err
// RUN: FileCheck < %t.err -check-prefix=CHECK-CODE-COMPLETE-CRASH %s
// RUN: test ! -e %t-preamble.pch
// CHECK-CODE-COMPLETE-CRASH: Unable to perform code completion!
//
// REQUIRES: crash-recovery

// FIXME: Please investigate abnormal path in MemoryBuffer.
// REQUIRES: can-remove-opened-file
// UNSUPPORTED: libstdcxx-safe-mode

#warning parsing original file
