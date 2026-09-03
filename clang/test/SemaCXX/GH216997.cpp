// RUN: %clang_cc1 -triple x86_64 -std=c++20 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64 -std=c++20 -fsyntax-only -verify -fexperimental-new-constant-interpreter %s

// Reproducer from GH216997.
using vec __attribute__((vector_size(16))) = int &bar; // expected-error {{type-id cannot have a name}}
int baz = __builtin_vectorelements(vec); // expected-error {{argument to __builtin_vectorelements must be of vector type}}

using vec_ref __attribute__((vector_size(16))) = int &;
static_assert(sizeof(vec_ref) == 16, "");
int a = __builtin_vectorelements(vec_ref); // expected-error {{argument to __builtin_vectorelements must be of vector type}}

typedef int veci4 __attribute__((vector_size(16)));
int b = __builtin_vectorelements(veci4 &); // expected-error {{argument to __builtin_vectorelements must be of vector type}}
int c = __builtin_vectorelements(veci4 &&); // expected-error {{argument to __builtin_vectorelements must be of vector type}}
int d = __builtin_vectorelements(const veci4 &); // expected-error {{argument to __builtin_vectorelements must be of vector type}}

veci4 v;
int e = __builtin_vectorelements(decltype((v))); // expected-error {{argument to __builtin_vectorelements must be of vector type}}

template <typename T>
int f() {
  return __builtin_vectorelements(T); // expected-error {{argument to __builtin_vectorelements must be of vector type}}
}
int g = f<veci4>();
int h = f<veci4 &>(); // expected-note {{in instantiation of function template specialization}}

void ok(veci4 &r, veci4 &&rr) {
  (void)__builtin_vectorelements(r);
  (void)__builtin_vectorelements(rr);
  (void)__builtin_vectorelements(const veci4);
  (void)__builtin_vectorelements(decltype(v));
}
