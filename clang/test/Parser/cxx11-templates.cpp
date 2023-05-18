// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

struct S {
  template <typename Ty = char>
  static_assert(sizeof(Ty) != 1, "Not a char"); // expected-error {{a static_assert declaration cannot be a template}}
};

template <typename Ty = char>
static_assert(sizeof(Ty) != 1, "Not a char"); // expected-error {{a static_assert declaration cannot be a template}}

namespace Ellipsis {
  template<typename ...T> void f(T t..., int n); // expected-error {{must immediately precede declared identifier}}
  template<typename ...T> void f(int n, T t...); // expected-error {{must immediately precede declared identifier}}
  template<typename ...T> void f(int n, T t, ...); // expected-error {{unexpanded parameter pack}}
  template<typename ...T> void f() {
    f([]{
      void g(T
             t // expected-note {{place '...' immediately before declared identifier to declare a function parameter pack}}
             ... // expected-warning {{'...' in this location creates a C-style varargs function, not a function parameter pack}}
             // expected-note@-1 {{insert ',' before '...' to silence this warning}}
             );
      void h(T (&
              ) // expected-note {{place '...' here to declare a function parameter pack}}
             ... // expected-warning {{'...' in this location creates a C-style varargs function, not a function parameter pack}}
             // expected-note@-1 {{insert ',' before '...' to silence this warning}}
             );
      void i(T (&), ...);
    }...);
  }
  template<typename ...T> struct S {
    void f(T t...); // expected-error {{must immediately precede declared identifier}}
    void f(T ... // expected-note {{preceding '...' declares a function parameter pack}}
           t...); // expected-warning-re {{'...' in this location creates a C-style varargs function{{$}}}}
           // expected-note@-1 {{insert ',' before '...' to silence this warning}}
  };

  // FIXME: We should just issue a single error in this case pointing out where
  // the '...' goes. It's tricky to recover correctly in this case, though,
  // because the parameter is in scope in the default argument, so must be
  // passed to Sema before we reach the ellipsis.
  template<typename...T> void f(T n = 1 ...);
  // expected-warning@-1 {{creates a C-style varargs}}
  // expected-note@-2 {{place '...' immediately before declared identifier}}
  // expected-note@-3 {{insert ','}}
  // expected-error@-4 {{unexpanded parameter pack}}
}
