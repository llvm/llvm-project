// Don't crash.

// RUN: cp %S/modified-header-crash.h %t.h
// RUN: %clang_cc1 -DCAKE -x c-header %t.h -emit-pch -o %t
// RUN: echo >> %t.h
// RUN: not %clang_cc1 %s -include-pch %t -fsyntax-only

void f(void) {
  foo = 3;
}
