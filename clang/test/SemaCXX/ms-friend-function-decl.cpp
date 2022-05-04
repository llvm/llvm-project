// RUN: %clang_cc1 -std=c++03 -fms-compatibility -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++17 -fms-compatibility -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++20 -fms-compatibility -fsyntax-only -verify=modern %s
#if __cplusplus < 202002L
// expected-no-diagnostics
#endif

namespace ns {

class C {
public:
  template <typename T>
  friend void funtemp();

  friend void fun();

  void test() {
    ::ns::fun(); // modern-error {{no member named 'fun' in namespace 'ns'}}

    // modern-error@+3 {{no member named 'funtemp' in namespace 'ns'}}
    // modern-error@+2 {{expected '(' for function-style cast or type construction}}
    // modern-error@+1 {{expected expression}}
    ::ns::funtemp<int>();
  }
};

void fun() {
}

template <typename T>
void funtemp() {}

} // namespace ns

class Glob {
public:
  friend void funGlob();

  void test() {
    funGlob(); // modern-error {{use of undeclared identifier 'funGlob'}}
  }
};

void funGlob() {
}
