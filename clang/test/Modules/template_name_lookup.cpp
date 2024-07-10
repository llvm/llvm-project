// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: %clang_cc1 -std=c++20 %S/Inputs/template_name_lookup/foo.cppm -emit-module-interface -o %t/foo.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %s -fsyntax-only -verify

import foo;
void use() {
  X x; // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'X'}}
       // expected-note@Inputs/template_name_lookup/foo.cppm:3 {{candidate template ignored: couldn't infer template argument 'T'}}
       // expected-note@Inputs/template_name_lookup/foo.cppm:3 {{implicit deduction guide declared as 'template <typename T> X(X<T>) -> X<T>'}}
       // expected-note@Inputs/template_name_lookup/foo.cppm:3 {{candidate function template not viable: requires 1 argument, but 0 were provided}}
       // expected-note@Inputs/template_name_lookup/foo.cppm:3 {{implicit deduction guide declared as 'template <typename T> X() -> X<T>'}}
}
