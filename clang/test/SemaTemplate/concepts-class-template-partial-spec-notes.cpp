// RUN: %clang_cc1 -std=c++20 -verify %s

template <typename T>
concept integral = __is_integral(T); // expected-note 5 {{because '__is_integral(double)' evaluated to false}}

// Case 1: Issue #238212 reproducer (static_assert failure with concept constraint)
namespace case1 {
  template <typename T>
  struct trait {
    static constexpr bool value = false;
  };

  template <integral T> // expected-note {{because 'double' does not satisfy 'integral'}}
  struct trait<T> { // expected-note {{candidate template ignored: constraints not satisfied [with T = double]}}
    static constexpr bool value = true;
  };

  static_assert(trait<int>::value); // OK, successful compilation has no diagnostics
  static_assert(trait<double>::value, "expected integral"); // expected-error {{static assertion failed due to requirement 'case1::trait<double>::value': expected integral}}
}

// Case 2: requires-clause with unsatisfied atomic constraint
namespace case2 {
  template <typename T>
  struct trait {
    static constexpr bool value = false;
  };

  template <typename T>
  requires (sizeof(T) == 4) // expected-note {{because 'sizeof(double) == 4' (8 == 4) evaluated to false}}
  struct trait<T> { // expected-note {{candidate template ignored: constraints not satisfied [with T = double]}}
    static constexpr bool value = true;
  };

  static_assert(trait<double>::value); // expected-error {{static assertion failed due to requirement 'case2::trait<double>::value'}}
}

// Case 3: Undefined primary template
namespace case3 {
  template <typename T>
  struct trait; // expected-note {{template is declared here}}

  template <integral T> // expected-note {{because 'double' does not satisfy 'integral'}}
  struct trait<T> { // expected-note {{candidate template ignored: constraints not satisfied [with T = double]}}
    using type = int;
  };

  trait<double> t; // expected-error {{implicit instantiation of undefined template 'case3::trait<double>'}}
}

// Case 4: Missing member in primary template
namespace case4 {
  template <typename T>
  struct trait {};

  template <integral T> // expected-note {{because 'double' does not satisfy 'integral'}}
  struct trait<T> { // expected-note {{candidate template ignored: constraints not satisfied [with T = double]}}
    using type = int;
  };

  typename trait<double>::type x; // expected-error {{no type named 'type' in 'case4::trait<double>'}}
  typename trait<int>::type y; // OK
}

// Case 5: Multiple constrained partial specializations
namespace case5 {
  template <typename T>
  struct trait {
    static constexpr bool value = false;
  };

  template <typename T>
  requires (sizeof(T) == 1) // expected-note {{because 'sizeof(double) == 1' (8 == 1) evaluated to false}}
  struct trait<T> { // expected-note {{candidate template ignored: constraints not satisfied [with T = double]}}
    static constexpr bool value = true;
  };

  template <integral T> // expected-note {{because 'double' does not satisfy 'integral'}}
  struct trait<T> { // expected-note {{candidate template ignored: constraints not satisfied [with T = double]}}
    static constexpr bool value = true;
  };

  static_assert(trait<double>::value); // expected-error {{static assertion failed due to requirement 'case5::trait<double>::value'}}
}

// Case 6: Variable template partial specialization
namespace case6 {
  template <typename T>
  constexpr bool trait_v = false;

  template <integral T> // expected-note {{because 'double' does not satisfy 'integral'}}
  constexpr bool trait_v<T> = true; // expected-note {{candidate template ignored: constraints not satisfied [with T = double]}}

  static_assert(trait_v<int>); // OK
  static_assert(trait_v<double>); // expected-error {{static assertion failed due to requirement 'trait_v<double>'}}
}

// Case 7: Partial ordering - candidate rejected due to partial ordering rather than constraints
namespace case7 {
  template <typename T>
  concept C1 = sizeof(T) >= 4;

  template <typename T>
  concept C2 = C1<T> && sizeof(T) <= 8;

  template <typename T>
  struct trait {
    static constexpr int value = 0;
  };

  template <C1 T>
  struct trait<T> { // expected-note {{candidate is less constrained than the selected specialization}}
    static constexpr int value = 1;
  };

  template <C2 T>
  struct trait<T> {
    static constexpr int value = 2;
  };

  static_assert(trait<int>::value == 1); // expected-error {{static assertion failed due to requirement 'case7::trait<int>::value == 1'}} \
                                         // expected-note {{expression evaluates to '2 == 1'}}
}
