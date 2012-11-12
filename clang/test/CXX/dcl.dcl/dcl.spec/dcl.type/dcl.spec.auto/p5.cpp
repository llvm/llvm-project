// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify %s -std=c++11

struct S {
  virtual ~S();

  auto a; // expected-error{{'auto' not allowed in non-static struct member}}
  auto *b; // expected-error{{'auto' not allowed in non-static struct member}}
  const auto c; // expected-error{{'auto' not allowed in non-static struct member}}

  void f() throw (auto); // expected-error{{'auto' not allowed here}}

  friend auto; // expected-error{{'auto' not allowed in non-static struct member}}

  operator auto(); // expected-error{{'auto' not allowed here}}
};

// PR 9278: auto is not allowed in typedefs, except with a trailing return type.
typedef auto *AutoPtr; // expected-error{{'auto' not allowed in typedef}}
typedef auto (*PFun)(int a); // expected-error{{'auto' not allowed in typedef}}
typedef auto Fun(int a) -> decltype(a + a);

void g(auto a) { // expected-error{{'auto' not allowed in function prototype}}
  try { }
  catch (auto &a) { } // expected-error{{'auto' not allowed in exception declaration}}
  catch (const auto a) { } // expected-error{{'auto' not allowed in exception declaration}}
  try { } catch (auto a) { } // expected-error{{'auto' not allowed in exception declaration}}
}

void h(auto a[10]) { // expected-error{{'auto' not allowed in function prototype}}
}

void i(const auto a) { // expected-error{{'auto' not allowed in function prototype}}
}

namespace std {
  class type_info;
}

template<typename T> struct U {};

void j() {
  (void)typeid(auto); // expected-error{{'auto' not allowed here}}
  (void)sizeof(auto); // expected-error{{'auto' not allowed here}}
  (void)__alignof(auto); // expected-error{{'auto' not allowed here}}

  U<auto> v; // expected-error{{'auto' not allowed in template argument}}

  int n;
  (void)dynamic_cast<auto&>(n); // expected-error{{'auto' not allowed here}}
  (void)static_cast<auto*>(&n); // expected-error{{'auto' not allowed here}}
  (void)reinterpret_cast<auto*>(&n); // expected-error{{'auto' not allowed here}}
  (void)const_cast<auto>(n); // expected-error{{'auto' not allowed here}}
  (void)*(auto*)(&n); // expected-error{{'auto' not allowed here}}
  (void)auto(n); // expected-error{{expected expression}}
  (void)auto{n}; // expected-error{{expected expression}}
}

template <auto a = 10> class C { }; // expected-error{{'auto' not allowed in template parameter}}
int ints[] = {1, 2, 3};
template <const auto (*a)[3] = &ints> class D { }; // expected-error{{'auto' not allowed in template parameter}}
enum E : auto {}; // expected-error{{'auto' not allowed here}}
struct F : auto {}; // expected-error{{expected class name}}
template<typename T = auto> struct G { }; // expected-error{{'auto' not allowed in template argument}}

using A = auto; // expected-error{{'auto' not allowed in type alias}}

auto k() -> auto; // expected-error{{'auto' not allowed in function return type}}
