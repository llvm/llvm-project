// RUN: %clang -S -emit-llvm %s &>%t.err; FileCheck %s --input-file=%t.err

#include <ripple.h>

// CHECK:      no-fenable-ripple.c:3:10: fatal error: 'ripple.h' file not found
// CHECK-NEXT:     3 | #include <ripple.h>
// CHECK-NEXT:       |          ^~~~~~~~~~
// CHECK-NEXT: 1 error generated.