// RUN: %clang_cc1 -x objective-c %s -fsyntax-only -verify
// expected-no-diagnostics

typedef __typeof__(((int*)0)-((int*)0)) ptrdiff_t;
extern id objc_assign_ivar(id value, id dest, ptrdiff_t offset);

