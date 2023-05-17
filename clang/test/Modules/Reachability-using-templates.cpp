// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/mod.templates.cppm -emit-module-interface -o %t/mod.templates.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -fsyntax-only -verify
//
//--- mod.templates.cppm
export module mod.templates;
template <class> struct t {};
export template <class T> using u = t<T>;

//--- Use.cpp
// expected-no-diagnostics
import mod.templates;
void foo() {
  u<int> v{};
}
