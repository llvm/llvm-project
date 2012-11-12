// RUN: rm -f %t.c
// RUN: touch %t.c
// RUN: c-index-test -write-pch %t.pch %t.c
// RUN: cp %s %t.c
// RUN: c-index-test -test-load-tu %t.pch local 2>&1 | FileCheck %s

// rdar://11558355
// Unfortunately this would crash reliably only via valgrind.

// CHECK: Unable to load translation unit
