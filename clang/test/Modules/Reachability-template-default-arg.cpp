// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/template_default_arg.cppm -emit-module-interface -o %t/template_default_arg.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -fsyntax-only -verify

// RUN: %clang_cc1 -std=c++20 %t/template_default_arg.cppm -emit-reduced-module-interface -o %t/template_default_arg.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -fsyntax-only -verify
//
//--- template_default_arg.cppm
export module template_default_arg;
struct t {};

export template <typename T = t>
struct A {
  T a;
};

//--- Use.cpp
import template_default_arg;
void bar() {
  A<> a0;
  A<t> a1; // expected-error {{use of undeclared identifier 't'}}
}
