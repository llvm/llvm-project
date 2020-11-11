// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

template <typename T, typename U = void*>
struct A {
  enum {
    id = _Generic(T(), // expected-error {{controlling expression type 'char' not compatible with any generic association type}}
        int: 1, // expected-note {{compatible type 'int' specified here}}
        float: 2,
        U: 3) // expected-error {{type 'int' in generic association compatible with previously specified type 'int'}}
  };
};

static_assert(A<int>::id == 1, "fail");
static_assert(A<float>::id == 2, "fail");
static_assert(A<double, double>::id == 3, "fail");

A<char> a1; // expected-note {{in instantiation of template class 'A<char>' requested here}}
A<short, int> a2; // expected-note {{in instantiation of template class 'A<short, int>' requested here}}

template <typename T, typename U>
struct B {
  enum {
    id = _Generic(T(),
        int: 1, // expected-note {{compatible type 'int' specified here}}
        int: 2, // expected-error {{type 'int' in generic association compatible with previously specified type 'int'}}
        U: 3)
  };
};

template <unsigned Arg, unsigned... Args> struct Or {
  enum { result = Arg | Or<Args...>::result };
};

template <unsigned Arg> struct Or<Arg> {
  enum { result = Arg };
};

template <class... Args> struct TypeMask {
  enum {
   result = Or<_Generic(Args(), int: 1, long: 2, short: 4, float: 8)...>::result
  };
};

static_assert(TypeMask<int, long, short>::result == 7, "fail");
static_assert(TypeMask<float, short>::result == 12, "fail");
static_assert(TypeMask<int, float, float>::result == 9, "fail");
