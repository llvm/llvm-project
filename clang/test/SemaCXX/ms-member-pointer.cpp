// RUN: %clang_cc1 -triple x86_64-windows-msvc -verify -std=c++26 %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fdelayed-template-parsing -verify -std=c++26 %s
// expected-no-diagnostics

namespace dependent_complete {
  template <class> struct A {
    void f(void (A::*)()) {}
  };
} // namespace dependent_complete
