// RUN: %clang_cc1 -fms-compatibility -fsyntax-only -verify %s

struct C {
  template <typename T> static int foo(T);
};

template <typename T> static int C::foo(T) { 
  //expected-warning@-1 {{'static' can only be specified inside the class definition}}
  return 0;
}

