// RUN: not --crash %clang_cc1 %s 2>&1 | FileCheck %s
// REQUIRES: crash-recovery

// FIXME: CHECKs might be incompatible to win32.
// Stack traces also require back traces.
// REQUIRES: shell, backtrace

#prag\
ma clang __debug crash

// CHECK: prag\
// CHECK-NEXT: ma

