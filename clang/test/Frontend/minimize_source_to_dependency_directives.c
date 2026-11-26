// RUN: %clang_cc1 -print-dependency-directives-minimized-source %s > %t
// RUN: echo END. >> %t
// RUN: FileCheck < %t %s
// RUN: %clang_cc1 -Eonly %t

#ifdef FOO
#include "a.h"
#else
void skipThisCode();
#endif

// CHECK:      #ifdef FOO
// CHECK-NEXT: #include "a.h"
// CHECK-NEXT: #endif
// CHECK-NEXT: END.
