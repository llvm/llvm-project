// RUN: %check_clang_tidy -std=c++20 %s modernize-use-trailing-return-type %t

namespace std {
template <typename T, typename U>
struct is_same { static constexpr auto value = false; };

template <typename T>
struct is_same<T, T> { static constexpr auto value = true; };

template <typename T>
concept floating_point = std::is_same<T, float>::value || std::is_same<T, double>::value || std::is_same<T, long double>::value;
}

//
// Concepts
//

std::floating_point auto con1();
// CHECK-MESSAGES: :[[@LINE-1]]:26: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: {{^}}auto con1() -> std::floating_point auto;{{$}}

std::floating_point auto con1() { return 3.14f; }
// CHECK-MESSAGES: :[[@LINE-1]]:26: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: {{^}}auto con1() -> std::floating_point auto { return 3.14f; }{{$}}

namespace a {
template <typename T>
concept Concept = true;

template <typename T, typename U>
concept BinaryConcept = true;
}

a::Concept decltype(auto) con2();
// CHECK-MESSAGES: :[[@LINE-1]]:27: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: {{^}}auto con2() -> a::Concept decltype(auto);{{$}}

a::BinaryConcept<int> decltype(auto) con3();
// CHECK-MESSAGES: :[[@LINE-1]]:38: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: {{^}}auto con3() -> a::BinaryConcept<int> decltype(auto);{{$}}

const std::floating_point auto* volatile con4();
// CHECK-MESSAGES: :[[@LINE-1]]:42: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: {{^}}auto con4() -> const std::floating_point auto* volatile;{{$}}

template <typename T>
int req1(T t) requires std::floating_point<T>;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES: {{^}}auto req1(T t) -> int requires std::floating_point<T>;{{$}}

template <typename T>
T req2(T t) requires requires { t + t; };
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
  // CHECK-FIXES: {{^}}auto req2(T t) -> T requires requires { t + t; };{{$}}

//
// Operator c++20 defaulted comparison operators
//
// Requires <compare>

namespace std {
struct strong_ordering {
  using value_type = signed char;
  static strong_ordering const less;
  static strong_ordering const equal;
  static strong_ordering const equivalent;
  static strong_ordering const greater;

  constexpr strong_ordering(value_type v) : val(v) {}
  template <typename T>
  requires(T{0}) friend constexpr auto
  operator==(strong_ordering v, T u) noexcept -> bool {
    return v.val == u;
  }
  friend constexpr auto operator==(strong_ordering v, strong_ordering w) noexcept -> bool = default;

  value_type val{};
};
inline constexpr strong_ordering strong_ordering::less{-1};
inline constexpr strong_ordering strong_ordering::equal{0};
inline constexpr strong_ordering strong_ordering::equivalent{0};
inline constexpr strong_ordering strong_ordering::greater{1};

} // namespace std

struct TestDefaultOperatorA {
  int a{};
  int b{};

  friend auto operator<=>(const TestDefaultOperatorA &, const TestDefaultOperatorA &) noexcept = default;
};

struct TestDefaultOperatorB {
  int a{};
  int b{};
  friend auto operator==(const TestDefaultOperatorB &, const TestDefaultOperatorB &) noexcept -> bool = default;
  friend bool operator<(const TestDefaultOperatorB &, const TestDefaultOperatorB &) noexcept = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
  // CHECK-FIXES: {{^}}  friend auto operator<(const TestDefaultOperatorB &, const TestDefaultOperatorB &) noexcept -> bool = default;{{$}}
};
