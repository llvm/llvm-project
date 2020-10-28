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
