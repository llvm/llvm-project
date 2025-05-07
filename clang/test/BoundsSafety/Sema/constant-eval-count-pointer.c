
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s

#include <ptrcheck.h>

void foo(long len, int *buf __counted_by(len)) {
// This is a reduced test that exercises BoundsSafetyPointerCast
// in ExprConstant's pointer constant evaluation when source
// pointer type is counted, invoked from pointer arithmetic
// overflow check.
    buf + 0;
}
