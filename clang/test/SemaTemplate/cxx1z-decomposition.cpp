// RUN: %clang_cc1 -std=c++1z -verify %s

struct A { int x, y; };
typedef int B[2];
struct C { template<int> int get(); };
struct D { int x, y, z; };
struct E { int *p, n; };

namespace std {
  using size_t = decltype(sizeof(0));
  template<typename> struct tuple_size;
  template<size_t, typename> struct tuple_element { using type = int; };
}

template<> struct std::tuple_size<C> { enum { value = 2 }; };

template<typename T> int decomp(T &t) { 
  auto &[a, b] = t; // expected-error {{type 'D' binds to 3 elements, but only 2 names were provided}}
  return a + b; // expected-error {{cannot initialize return object of type 'int' with an rvalue of type 'int *'}}
}

void test() {
  A a;
  B b;
  C c;
  D d;
  E e;
  decomp(a);
  decomp(b);
  decomp(c);
  decomp(d); // expected-note {{in instantiation of}}
  decomp(e); // expected-note {{in instantiation of}}
}
