// RUN: %clang_dxc -DTEST=2 -Tlib_6_7 -### %s 2>&1 | FileCheck %s

// Make sure -D send to cc1.
// CHECK:"-D" "TEST=2"

#ifndef TEST
#error "TEST not defined"
#elif TEST != 2
#error "TEST defined to wrong value"
#endif
