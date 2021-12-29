// RUN: %clang_cc1 -std=c++20 -verify %s
// RUN: %clang_cc1 -std=c++20 -verify %s -DDEPENDENT_OR

#ifdef DEPENDENT_OR
// This causes the || below to be a CXXOperatorCallExpr not a BinaryOperator.
struct A {}; bool operator||(A, A);
#endif

namespace PR45589 {
  template<typename T> struct X { static constexpr bool value = T::value; }; // expected-error {{cannot be used prior to '::'}}
  struct False { static constexpr bool value = false; };
  struct True { static constexpr bool value = true; };

  template<typename T> concept C = true;

  template<bool B, typename T> constexpr int test = 0;
  template<bool B, typename T> requires C<T> constexpr int test<B, T> = 1;
  template<bool B, typename T> requires (B && C<T>) || (X<T>::value && C<T>) constexpr int test<B, T> = 2; // expected-error {{non-constant expression}} expected-note {{subexpression}} expected-note {{instantiation of}} expected-note {{while substituting}}
  static_assert(test<true, False> == 2);
  static_assert(test<true, True> == 2);
  static_assert(test<true, char> == 2); // satisfaction of second term of || not considered
  static_assert(test<false, False> == 1);
  static_assert(test<false, True> == 2); // constraints are partially ordered
  // FIXME: These diagnostics are excessive.
  static_assert(test<false, char> == 1); // expected-note 2{{while}} expected-note 2{{during}}
}

namespace PR52905 {
// A mock for std::convertible_to. Not complete support.
template <typename _From, typename _To>
concept convertible_to = __is_convertible_to(_From, _To); // expected-note {{evaluated to false}}

template <typename T>
class A {
public:
  using iterator = void **;

  iterator begin();
  const iterator begin() const;
};

template <class T>
concept Beginable1 = requires(T t) {
  { t.begin }
  ->convertible_to<typename T::iterator>; // expected-note {{not satisfied}}
};

static_assert(Beginable1<A<int>>); // expected-error {{static_assert failed}}
                                   // expected-note@-1 {{does not satisfy 'Beginable1'}}

template <class T>
concept Beginable2 = requires(T t) {
  { t.begin() }
  ->convertible_to<typename T::iterator>;
};

static_assert(Beginable2<A<int>>);
} // namespace PR52905
