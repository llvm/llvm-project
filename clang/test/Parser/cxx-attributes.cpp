// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

class c {
  virtual void f1(const char* a, ...)
    __attribute__ (( __format__(__printf__,2,3) )) = 0;
  virtual void f2(const char* a, ...)
    __attribute__ (( __format__(__printf__,2,3) )) {}
};

