// Test CF+LF are properly handled along with quoted, multi-line #error
// RUN: %clang_cc1 -DOTHER -print-dependency-directives-minimized-source %s 2>&1 | FileCheck %s

#ifndef TEST
#error "message \
   more message \
   even more"
#endif

#ifdef OTHER
#include <string>
#endif

// CHECK:      #ifdef OTHER
// CHECK-NEXT: #include <string>
// CHECK-NEXT: #endif
