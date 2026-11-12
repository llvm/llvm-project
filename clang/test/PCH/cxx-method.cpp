// RUN: %clang_cc1 -x c++ -include %S/Inputs/cxx-method.h -verify %s
// RUN: %clang_cc1 -x c++ -emit-pch %S/Inputs/cxx-method.h -o %t
// RUN: %clang_cc1 -include-pch %t -verify %s
// expected-no-diagnostics

void S::m(int x) { }

S::operator char *() { return 0; }

S::operator const char *() { return 0; }

struct T : S {};

const T a = T();
T b(a);
