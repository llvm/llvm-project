// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_PREAMBLE_FILE=%t-preamble.pch \
// RUN:   not c-index-test -code-completion-at=%s:20:1 \
// RUN:   "-remap-file=%s;%S/Inputs/crash-recovery-code-complete-remap.c" \
// RUN:   %s 2> %t.err
// RUN: FileCheck < %t.err -check-prefix=CHECK-CODE-COMPLETE-CRASH %s
// RUN: test ! -e %t-preamble.pch
// CHECK-CODE-COMPLETE-CRASH: Unable to perform code completion!
//
// REQUIRES: crash-recovery
// REQUIRES: shell

#warning parsing original file
