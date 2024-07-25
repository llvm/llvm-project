// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fenable-matrix -verify %s
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -fenable-matrix -verify %s

template<typename T>
void f(T dependent) {
  int i;
  i = {};
  i = { dependent };
  i = { dependent, dependent };
}

template<typename T>
void f2(T dependent) {
  int i;
  i = {};
  i = { dependent };
  i = { dependent, dependent }; // expected-error {{excess elements in scalar initializer}}
}
template void f2(int); // expected-note {{in instantiation of function template specialization 'f2<int>' requested here}}

void g() {
  int i;
  i = {};
  i = { 0 };
  i = { int{} };
  i = { {} }; // expected-warning {{too many braces around scalar initializer}}
  i = { { 0 } }; // expected-warning {{too many braces around scalar initializer}}
  i += { 0 }; // expected-error {{initializer list cannot be used on the right hand side of operator '+'}}

  auto np = nullptr;
  np = {};
  np = { nullptr };
  np = { 0 };
  np = { 1 }; // expected-error {{cannot initialize a value of type 'std::nullptr_t' with an rvalue of type 'int'}}

  void* vp;
  vp = {};
  vp = { (void*)nullptr };
  vp = { nullptr };
  vp = { (int*)nullptr };
  vp = { 0 };

  const void* arr[1] = { nullptr };
  arr = { nullptr }; // expected-error {{array type 'const void *[1]' is not assignable}}
  arr = { arr }; // expected-error {{array type 'const void *[1]' is not assignable}}
  arr = {}; // expected-error {{array type 'const void *[1]' is not assignable}}
  arr = { 1 }; // expected-error {{array type 'const void *[1]' is not assignable}}

  typedef int i1_t [[gnu::vector_size(sizeof(int))]];
  i1_t i1;
  i1 = {};
  i1 = { 0 };

  typedef int i4_t [[gnu::vector_size(4*sizeof(int))]];
  i4_t i4;
  i4 = {};
  i4 = { 0, 1, 2, 3 };

  // TODO: when matrix_type initialization is specified, try to
  // make these work/not work as required
#if 0
  typedef double [[clang::matrix_type(1,1)]] d11_t;
  d11_t d11;
  d11 = {};
  d11 = { 0.0 };
  d11 = { { 0.0 } };

  typedef double [[clang::matrix_type(2,2)]] d22_t;
  d22_t d22;
  d22 = {};
  d22 = { 0.0 };
  d22 = { { 0.0, 1.0 }, { 2.0, 3.0 } };
#endif

  double _Complex dc;
  dc = {};
  dc = { 0.0 };
  dc = { 0.0, 0.0 };
}
