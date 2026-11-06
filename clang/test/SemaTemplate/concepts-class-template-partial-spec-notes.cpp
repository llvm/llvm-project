// RUN: %clang_cc1 -std=c++20 -verify %s

template <typename T>
concept integral = __is_integral(T); // #integral_concept

// Case 1: Issue #238212 reproducer (static_assert failure with concept constraint)
namespace case1 {
  template <typename T>
  struct trait {
    static constexpr bool value = false;
  };

  template <integral T> // #case1_spec_decl
  struct trait<T> { // #case1_spec
    static constexpr bool value = true;
  };

  static_assert(trait<int>::value); // OK, successful compilation has no diagnostics
  static_assert(trait<double>::value, "expected integral"); // expected-error {{static assertion failed due to requirement 'case1::trait<double>::value': expected integral}} \
                                                            // expected-note@#case1_spec {{candidate template ignored: constraints not satisfied [with T = double]}} \
                                                            // expected-note@#case1_spec_decl {{because 'double' does not satisfy 'integral'}} \
                                                            // expected-note@#integral_concept {{because '__is_integral(double)' evaluated to false}}
}

// Case 2: requires-clause with unsatisfied atomic constraint
namespace case2 {
  template <typename T>
  struct trait {
    static constexpr bool value = false;
  };

  template <typename T>
  requires (sizeof(T) == 4) // #case2_requires
  struct trait<T> { // #case2_spec
    static constexpr bool value = true;
  };

  static_assert(trait<double>::value); // expected-error {{static assertion failed due to requirement 'case2::trait<double>::value'}} \
                                       // expected-note@#case2_spec {{candidate template ignored: constraints not satisfied [with T = double]}} \
                                       // expected-note@#case2_requires {{because 'sizeof(double) == 4' (8 == 4) evaluated to false}}
}

// Case 3: Undefined primary template
namespace case3 {
  template <typename T>
  struct trait; // #case3_primary

  template <integral T> // #case3_spec_decl
  struct trait<T> { // #case3_spec
    using type = int;
  };

  trait<double> t; // expected-error {{implicit instantiation of undefined template 'case3::trait<double>'}} \
                   // expected-note@#case3_primary {{template is declared here}} \
                   // expected-note@#case3_spec {{candidate template ignored: constraints not satisfied [with T = double]}} \
                   // expected-note@#case3_spec_decl {{because 'double' does not satisfy 'integral'}} \
                   // expected-note@#integral_concept {{because '__is_integral(double)' evaluated to false}}
}

// Case 4: Missing member in primary template
namespace case4 {
  template <typename T>
  struct trait {};

  template <integral T> // #case4_spec_decl
  struct trait<T> { // #case4_spec
    using type = int;
  };

  typename trait<double>::type x; // expected-error {{no type named 'type' in 'case4::trait<double>'}} \
                                  // expected-note@#case4_spec {{candidate template ignored: constraints not satisfied [with T = double]}} \
                                  // expected-note@#case4_spec_decl {{because 'double' does not satisfy 'integral'}} \
                                  // expected-note@#integral_concept {{because '__is_integral(double)' evaluated to false}}
  typename trait<int>::type y; // OK
}

// Case 5: Multiple constrained partial specializations
namespace case5 {
  template <typename T>
  struct trait {
    static constexpr bool value = false;
  };

  template <typename T>
  requires (sizeof(T) == 1) // #case5_req1
  struct trait<T> { // #case5_spec1
    static constexpr bool value = true;
  };

  template <integral T> // #case5_req2
  struct trait<T> { // #case5_spec2
    static constexpr bool value = true;
  };

  static_assert(trait<double>::value); // expected-error {{static assertion failed due to requirement 'case5::trait<double>::value'}} \
                                       // expected-note@#case5_spec1 {{candidate template ignored: constraints not satisfied [with T = double]}} \
                                       // expected-note@#case5_req1 {{because 'sizeof(double) == 1' (8 == 1) evaluated to false}} \
                                       // expected-note@#case5_spec2 {{candidate template ignored: constraints not satisfied [with T = double]}} \
                                       // expected-note@#case5_req2 {{because 'double' does not satisfy 'integral'}} \
                                       // expected-note@#integral_concept {{because '__is_integral(double)' evaluated to false}}
}

// Case 6: Variable template partial specialization
namespace case6 {
  template <typename T>
  constexpr bool trait_v = false;

  template <integral T> // #case6_spec_decl
  constexpr bool trait_v<T> = true; // #case6_spec

  static_assert(trait_v<int>); // OK
  static_assert(trait_v<double>); // expected-error {{static assertion failed due to requirement 'trait_v<double>'}} \
                                  // expected-note@#case6_spec {{candidate template ignored: constraints not satisfied [with T = double]}} \
                                  // expected-note@#case6_spec_decl {{because 'double' does not satisfy 'integral'}} \
                                  // expected-note@#integral_concept {{because '__is_integral(double)' evaluated to false}}
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
  struct trait<T> { // #case7_c1
    static constexpr int value = 1;
  };

  template <C2 T>
  struct trait<T> {
    static constexpr int value = 2;
  };

  static_assert(trait<int>::value == 1); // expected-error {{static assertion failed due to requirement 'case7::trait<int>::value == 1'}} \
                                         // expected-note {{expression evaluates to '2 == 1'}} \
                                         // expected-note@#case7_c1 {{candidate is less constrained than the selected specialization}}
}
