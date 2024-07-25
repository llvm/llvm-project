// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

template<typename T>
void f(T dependent) {
  int i;
  i = { dependent, dependent };
}

template<typename T>
void f2(T dependent) {
  int i;
  i = { dependent, dependent }; // expected-error {{excess elements in scalar initializer}}
}
template void f2(int); // expected-note {{in instantiation of function template specialization 'f2<int>' requested here}}

void g() {
  int i;
  i = {0};
  i += {0}; // expected-error {{initializer list cannot be used on the right hand side of operator '+'}}
}
