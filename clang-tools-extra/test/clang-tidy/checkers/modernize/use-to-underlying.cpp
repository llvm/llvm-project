// RUN: %check_clang_tidy -std=c++23 %s modernize-use-to-underlying %t

namespace std {
template<typename T>
constexpr auto to_underlying(T value) noexcept {
  return static_cast<__underlying_type(T)>(value);
}
}


enum class MyEnum { A = 1, B = 2 };

void test_basic_cast() {
  int value = static_cast<int>(MyEnum::A);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use 'std::to_underlying' instead of 'static_cast' for 'enum class' [modernize-use-to-underlying]
  // CHECK-FIXES: int value = std::to_underlying(MyEnum::A);
}


void test_long_cast() {
  long value = static_cast<long>(MyEnum::B);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use 'std::to_underlying' instead of 'static_cast' for 'enum class' [modernize-use-to-underlying]
  // CHECK-FIXES: long value = std::to_underlying(MyEnum::B);
}


void test_expression() {
  int result = static_cast<int>(MyEnum::A) + 10;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use 'std::to_underlying' instead of 'static_cast' for 'enum class' [modernize-use-to-underlying]
  // CHECK-FIXES: int result = std::to_underlying(MyEnum::A) + 10;
}

void test_already_correct() {
  int value = std::to_underlying(MyEnum::B);
  // No warning expected
}

void test_float_cast() {
  float y = 8.34;
  int z = static_cast<int>(y);
  // No warning expected
}
