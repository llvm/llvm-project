// RUN: %clang_cc1 -x c++-header -emit-pch %s -o %t
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -verify %s

// expected-no-diagnostics

#ifndef HEADER_INCLUDED
#define HEADER_INCLUDED

static inline void foo(int &x, int y) {
  // Capturing x and y
  #pragma clang __debug captured
  {
    x += y;
  }
}

struct C {
  int val;

  explicit C(int v) : val(v) { }

  void bar(int &x) {
    // Capturing x and this
    #pragma clang __debug captured
    {
      x += val;
    }
  }
};

#else

void test_foo(int &x) {
  foo(x, 10);
}

void test_bar(int &x) {
  C Obj(10);
  Obj.bar(x);
}

#endif
