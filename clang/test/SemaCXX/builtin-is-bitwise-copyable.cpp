// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

// Scalar types are bitwise copyable.
static_assert(__is_bitwise_copyable(int));
static_assert(__is_bitwise_copyable(int*));
// array
static_assert(__is_bitwise_copyable(int[10]));


struct Forward; // expected-note 2{{forward declaration of 'Forward'}}
static_assert(!__is_bitwise_copyable(Forward)); // expected-error {{incomplete type 'Forward' used in type trait expression}}

struct Foo { int a; };
static_assert(__is_bitwise_copyable(Foo));

struct DynamicClass { virtual int Foo(); };
static_assert(__is_bitwise_copyable(DynamicClass));

template <typename T>
void TemplateFunction() {
  static_assert(__is_bitwise_copyable(T)); // expected-error {{incomplete type 'Forward' used in type trait expression}}
}
void CallTemplateFunc() {
  TemplateFunction<Forward>(); // expected-note {{in instantiation of function template specialization}}
  TemplateFunction<Foo>();
}
