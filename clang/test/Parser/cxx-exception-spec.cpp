// RUN: %clang_cc1 -fsyntax-only  %s

struct X { };

struct Y { };

void f() throw() { }

void g(int) throw(X) { }

void h() throw(X, Y) { }

class Class {
  void foo() throw (X, Y) { }
};

void (*fptr)() throw();
