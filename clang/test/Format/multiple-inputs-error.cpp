// RUN: cp %s %t-1.cpp
// RUN: cp %s %t-2.cpp
// RUN: not clang-format 2>&1 >/dev/null -offset=1 -length=0 %t-1.cpp %t-2.cpp |FileCheck %s
// RUN: not clang-format 2>&1 >/dev/null -lines=1:1 %t-1.cpp %t-2.cpp |FileCheck %s -check-prefix=CHECK-LINE
// CHECK: error: -offset, -length and -lines can only be used for single file.
// CHECK-LINE: error: -offset, -length and -lines can only be used for single file.

int i ;
