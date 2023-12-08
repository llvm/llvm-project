// RUN: %clang_cc1 %s -fsyntax-only
// RUN: not %clang_cc1 %s -fsyntax-only -DFAIL 2>&1 | FileCheck %s

void foo(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(void (*f)(
#ifdef FAIL
void (*f)()
#endif
)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))));
// CHECK: fatal error: function scope depth exceeded maximum of 127
