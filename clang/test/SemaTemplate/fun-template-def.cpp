// RUN: %clang_cc1 -fsyntax-only -verify %s

// Tests that dependent expressions are always allowed, whereas non-dependent
// are checked as usual.

#include <stddef.h>

// Fake typeid, lacking a typeinfo header.
namespace std { class type_info {}; }

struct dummy {}; // expected-note 3 {{candidate constructor (the implicit copy constructor)}}

template<typename T>
int f0(T x) {
  return (sizeof(x) == sizeof(int))? 0 : (sizeof(x) == sizeof(double))? 1 : 2;
}

template <typename T, typename U>
T f1(T t1, U u1, int i1)
{
  T t2 = i1;
  t2 = i1 + u1;
  ++u1;
  u1++;
  int i2 = u1;

  i1 = t1[u1];
  i1 *= t1;

  i1(u1, t1); // error
  u1(i1, t1);

  U u2 = (T)i1;
  static_cast<void>(static_cast<U>(reinterpret_cast<T>(
    dynamic_cast<U>(const_cast<T>(i1)))));

  new U(i1, t1);
  new int(t1, u1);
  new (t1, u1) int;
  delete t1;

  dummy d1 = sizeof(t1); // expected-error {{no viable conversion}}
  dummy d2 = offsetof(T, foo); // expected-error {{no viable conversion}}
  dummy d3 = __alignof(u1); // expected-error {{no viable conversion}}
  i1 = typeid(t1); // expected-error {{assigning to 'int' from incompatible type 'const std::type_info'}}

  return u1;
}
