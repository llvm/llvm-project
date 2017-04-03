// RUN: %clang_cc1 -fsyntax-only -verify %s

struct param_num {
  void* Foo(int a) __attribute__((alloc_align(1))); // expected-error {{'alloc_align' attribute is invalid for the implicit this argument}}
};


template <typename T>
struct dependent_ret {
  T* Foo(int a) __attribute__((alloc_align(2)));// no-warning, ends up being int**.
  T Foo2(int a) __attribute__((alloc_align(2)));// expected-warning {{'alloc_align' attribute only applies to return values that are pointers or references}}
};

// Following 2 errors associated only with the 'float' versions below.
template <typename T>
struct dependent_param_struct {
  void* Foo(T param) __attribute__((alloc_align(2))); // expected-error {{'alloc_align' attribute argument may only refer to a function parameter of integer type}}
};

template <typename T>
void* dependent_param_func(T param) __attribute__((alloc_align(1)));// expected-error {{'alloc_align' attribute argument may only refer to a function parameter of integer type}}

template <int T>
void* illegal_align_param(int p) __attribute__((alloc_align(T))); // expected-error {{'alloc_align' attribute requires parameter 1 to be an integer constant}}

void dependent_impl() {
  dependent_ret<int> a; // expected-note {{in instantiation of template class 'dependent_ret<int>' requested here}}
  a.Foo(1);
  a.Foo2(1);
  dependent_ret<int*> b; 
  a.Foo(1);
  a.Foo2(1);

  dependent_param_struct<int> c; 
  c.Foo(1);
  dependent_param_struct<float> d; // expected-note {{in instantiation of template class 'dependent_param_struct<float>' requested here}}
  d.Foo(1.0);
  dependent_param_func<int>(1);
  dependent_param_func<float>(1); // expected-note {{in instantiation of function template specialization 'dependent_param_func<float>' requested here}}
}
