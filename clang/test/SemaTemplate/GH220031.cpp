// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

template <bool B>
struct EnableIf {};

template <>
struct EnableIf<true> {
  using type = int;
};

template <class T, T V>
struct IntegralConstant {
  static constexpr T value = V;
};

namespace NonTemplate {

template <class T>
// expected-error@+1 {{no member named 'flag'}}
struct Trait : IntegralConstant<bool, T::flag> {};

template <class D>
struct CRTP {
  // expected-note@+1 {{in instantiation of template class}}
  template <typename EnableIf<Trait<D>::value>::type = 0>
  void fn() const {}
};

// expected-note@+1 {{in instantiation of template class}}
struct Derived : CRTP<Derived> {
  static constexpr bool flag = true;
};

} // namespace NonTemplate

namespace Templated {

template <class T>
// expected-error@+1 {{no member named 'flag'}}
struct Trait : IntegralConstant<bool, T::flag> {};

template <class D>
struct CRTP {
  // expected-note@+1 {{in instantiation of template class}}
  template <typename EnableIf<Trait<D>::value>::type = 0>
  void fn() const {}
};

template <class T>
// expected-note@+1 {{in instantiation of template class}}
struct Derived : CRTP<Derived<T>> {};

// expected-note@+1 {{in instantiation of template class}}
Derived<void> derived;

} // namespace Templated

// Errors in a class body instantiated as a side effect of deduction are not in
// the immediate context and must not be treated as substitution failures.
namespace TargetOwnsSFINAE {

template <bool>
struct Holder {
  using type = int;
};

template <class T>
// expected-error@+1 {{no member named 'flag'}}
struct Trait : Holder<T::flag> {};

template <class T>
// expected-note@+1 {{in instantiation of template class}}
struct Bad : Trait<Bad<T>> {};

template <class T>
// expected-note@+1 {{in instantiation of template class}}
typename Bad<T>::type probe(int);

template <class>
char probe(...);

// expected-note@+1 {{while substituting explicitly-specified template}}
int x = sizeof(probe<int>(0));

} // namespace TargetOwnsSFINAE

// A lookup performed directly in a live SFINAE context must still select the
// fallback when the member is declared later in the class.
namespace LiveSFINAE {

using Yes = char[1];
using No = char[2];

template <class T, int = T::value>
No &probe(int);

template <class>
Yes &probe(...);

struct A {
  static_assert(sizeof(probe<A>(0)) == sizeof(Yes), "");
  static constexpr int value = 1;
};

} // namespace LiveSFINAE
