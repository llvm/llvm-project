// RUN: %clang_cc1 -std=c++2a -x c++ %s -verify

// Examples from standard

template<typename T, typename U>
concept convertible_to = requires(T t) { U(t); };

template<typename T>
concept R = requires (T i) {
  typename T::type;
  {*i} -> convertible_to<const typename T::type&>;
};

template<typename T> requires R<T> struct S {};

struct T {
  using type = int;
  type i;
  const type &operator*() { return i; }
};

using si = S<T>;

template<typename T>
requires requires (T x) { x + x; } // expected-note{{because 'x + x' would be invalid: invalid operands to binary expression ('T' and 'T')}}
T add(T a, T b) { return a + b; } // expected-note{{candidate template ignored: constraints not satisfied [with T = T]}}

int x = add(1, 2);
int y = add(T{}, T{}); // expected-error{{no matching function for call to 'add'}}

template<typename T>
concept C = requires (T x) { x + x; }; // expected-note{{because 'x + x' would be invalid: invalid operands to binary expression ('T' and 'T')}}
template<typename T> requires C<T> // expected-note{{because 'T' does not satisfy 'C'}}
T add2(T a, T b) { return a + b; } // expected-note{{candidate template ignored: constraints not satisfied [with T = T]}}

int z = add2(1, 2);
int w = add2(T{}, T{}); // expected-error{{no matching function for call to 'add2'}}