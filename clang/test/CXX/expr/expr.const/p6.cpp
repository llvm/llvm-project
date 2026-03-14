// RUN: %clang_cc1 -std=c++17 -verify %s

template<typename T> int not_constexpr() { return T::error; }
template<typename T> constexpr int is_constexpr() { return T::error; } // expected-error {{'::'}}

template<typename T> int not_constexpr_var = T::error;
template<typename T> constexpr int is_constexpr_var = T::error; // expected-error {{'::'}}
template<typename T> const int is_const_var = T::error; // expected-error {{'::'}}
template<typename T> const volatile int is_const_volatile_var = T::error;
template<typename T> T is_dependent_var = T::error; // expected-error {{'::'}}
template<typename T> int &is_reference_var = T::error; // expected-error {{'::'}}
template<typename T> float is_float_var = T::error;

void test() {
  // Do not instantiate functions referenced in unevaluated operands...
  (void)sizeof(not_constexpr<long>());
  (void)sizeof(is_constexpr<long>());
  (void)sizeof(not_constexpr_var<long>);
  (void)sizeof(is_constexpr_var<long>);
  (void)sizeof(is_const_var<long>);
  (void)sizeof(is_const_volatile_var<long>);
  (void)sizeof(is_dependent_var<long>);
  (void)sizeof(is_dependent_var<const long>);
  (void)sizeof(is_reference_var<long>);
  (void)sizeof(is_float_var<long>);

  // ... but do if they are potentially constant evaluated, and refer to
  // constexpr functions or to variables usable in constant expressions.
  (void)sizeof(int{not_constexpr<int>()});
  (void)sizeof(int{is_constexpr<int>()}); // expected-note {{instantiation of}}
  (void)sizeof(int{not_constexpr_var<int>});
  (void)sizeof(int{is_constexpr_var<int>}); // expected-note {{instantiation of}}
  (void)sizeof(int{is_const_var<int>}); // expected-note {{instantiation of}}
  (void)sizeof(int{is_const_volatile_var<int>});
  (void)sizeof(int{is_dependent_var<int>});
  (void)sizeof(int{is_dependent_var<const int>}); // expected-note {{instantiation of}}
  (void)sizeof(int{is_reference_var<int>}); // expected-note {{instantiation of}}
  (void)sizeof(int{is_float_var<int>}); // expected-error {{cannot be narrowed}} expected-note {{cast}}
}
