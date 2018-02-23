// RUN: cp %s %t
// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: not %clang_cc1 -x c++ -fixit %t -Werror -DFIXIT
// RUN: %clang_cc1 -x c++ %t -DFIXIT
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++1z -Wc++14-compat

namespace foo1::foo2::foo3 {
#if __cplusplus <= 201400L
// expected-warning@-2 {{nested namespace definition is a C++1z extension; define each namespace separately}}
#else
// expected-warning@-4 {{nested namespace definition is incompatible with C++ standards before C++1z}}
#endif
  int foo(int x) { return x; }
}

#ifndef FIXIT
inline namespace goo::bar { // expected-error {{nested namespace definition cannot be 'inline'}} expected-warning 0-1{{C++11 feature}}
  int n;
}

int m = goo::bar::n;
#endif

int foo(int x) {
  return foo1::foo2::foo3::foo(x);
}

namespace bar1 {
  namespace bar2 {
    namespace bar3 {
      int bar(int x) { return x; }
    }
  }
}

int bar(int x) {
  return bar1::bar2::bar3::bar(x);
}
