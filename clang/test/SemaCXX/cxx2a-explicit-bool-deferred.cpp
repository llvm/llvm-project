// RUN: %clang_cc1 -fsyntax-only -verify -std=c++2a %s

template <typename T1, typename T2> struct is_same {
  static constexpr bool value = false;
};

template <typename T> struct is_same<T, T> {
  static constexpr bool value = true;
};

template <class T, class U>
concept SameHelper = is_same<T, U>::value;
template <class T, class U>
concept same_as = SameHelper<T, U> && SameHelper<U, T>;

namespace deferred_instantiation {
template <class X> constexpr X do_not_instantiate() { return nullptr; }

struct T {
  template <same_as<float> X> explicit(do_not_instantiate<X>()) T(X) {}

  T(int) {}
};

T t(5);
// expected-error@17{{cannot initialize}}
// expected-note@20{{in instantiation of function template specialization}}
// expected-note@30{{while substituting deduced template arguments}}
// expected-note@30{{in instantiation of function template specialization}}
T t2(5.0f);
} // namespace deferred_instantiation
