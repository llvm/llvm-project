// Tests that the lookup in transparent declaration context
// (linkage specifiaction context) wouldn't cause compiler crash.
// RUN: %clang_cc1 -std=c++20 %s -fsyntax-only -verify
extern "C++" {
template <class T>
class X {}; // expected-note {{candidate template ignored: couldn't infer template argument 'T'}} \
            // expected-note {{implicit deduction guide declared as 'template <class T> X(X<T>) -> X<T>'}} \
            // expected-note {{candidate function template not viable: requires 1 argument, but 0 were provided}} \
            // expected-note {{implicit deduction guide declared as 'template <class T> X() -> X<T>'}}
}

void foo() {
  X x; // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'X'}}
}
