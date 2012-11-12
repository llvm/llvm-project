// RUN: echo env CINDEXTEST_EDITING=1 \
// RUN:   not c-index-test -test-load-source-reparse 1 local \
// RUN:   -remap-file="%s;%S/Inputs/crash-recovery-code-complete-remap.c" \
// RUN:   %s 2> %t.err
// RUN: FileCheck < %t.err -check-prefix=CHECK-CODE-COMPLETE-CRASH %s
// CHECK-CODE-COMPLETE-CRASH: Unable to reparse translation unit
//
// XFAIL: win32

#warning parsing original file

#pragma clang __debug crash
