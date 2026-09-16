// RUN: %clang_cc1 -x c++ -include %S/Inputs/cxx-method.h -verify %s
// RUN: %clang_cc1 -x c++ -emit-pch %S/Inputs/cxx-method.h -o %t
// RUN: %clang_cc1 -include-pch %t -verify %s -error-on-deserialized-decl doNotDeserialize -ast-dump
// expected-no-diagnostics

// Calling constructors should not cause doNotDeserialize to be deserialized.
S s;
S s2(s);

// Implicitly defining special member functions should not cause
// doNotDeserialize to be deserialized.
Trivial t;
Trivial t2(t);

void assign() {
  s = s2;
  t = t2;
}

void S::m(int x) { }

S::operator char *() { return 0; }

S::operator const char *() { return 0; }

struct T : S {};

const T a = T();
T b(a);