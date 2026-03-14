// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR7463: Make sure that when we have an rvalue, it does not have
// cv-qualified non-class type.
template <typename T_> void g (T_&); // expected-note 7{{not viable}}

template<const int X> void h() { 
  g(X); // expected-error{{no matching function for call to 'g'}}
}

template<typename T, T X> void h2() { 
  g(X); // expected-error{{no matching function for call to 'g'}}
}

void a(__builtin_va_list x) {
  g(__builtin_va_arg(x, const int)); // expected-error{{no matching function for call to 'g'}}
  g((const int)0); // expected-error{{no matching function for call to 'g'}}
  typedef const int cint;
  g(cint(0)); // expected-error{{no matching function for call to 'g'}}
  g(static_cast<const int>(1)); // expected-error{{no matching function for call to 'g'}}
  g(reinterpret_cast<int *const>(0)); // expected-error{{no matching function for call to 'g'}}
  h<0>(); 
  h2<const int, 0>(); // expected-note{{instantiation of}}
}
