// RUN: %check_clang_tidy -std=c++23 %s modernize-use-to-underlying %t

// Mock std::to_underlying for testing
namespace std {
template<typename T>
constexpr auto to_underlying(T value) noexcept {
  return static_cast<__underlying_type(T)>(value);
}
}

// Test case 1: Basic enum class to int cast - should warn
enum class MyEnum { A = 1, B = 2 };

void test_basic_cast() {
  int value = static_cast<int>(MyEnum::A);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use 'std::to_underlying' instead of 'static_cast' for 'enum class' [modernize-use-to-underlying]
  // CHECK-FIXES: int value = std::to_underlying(MyEnum::A);
}

// Test case 2: enum class to long cast - should warn
void test_long_cast() {
  long value = static_cast<long>(MyEnum::B);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use 'std::to_underlying' instead of 'static_cast' for 'enum class' [modernize-use-to-underlying]
  // CHECK-FIXES: long value = std::to_underlying(MyEnum::B);
}

// Test case 3: enum class in expression - should warn
void test_expression() {
  int result = static_cast<int>(MyEnum::A) + 10;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use 'std::to_underlying' instead of 'static_cast' for 'enum class' [modernize-use-to-underlying]
  // CHECK-FIXES: int result = std::to_underlying(MyEnum::A) + 10;
}

// Test case 4: Already using std::to_underlying - should NOT warn
void test_already_correct() {
  int value = std::to_underlying(MyEnum::B);
  // No warning expected
}

// Test case 5: Casting float to int - should NOT warn (not an enum)
void test_float_cast() {
  float y = 8.34;
  int z = static_cast<int>(y);
  // No warning expected
}

// Test case 6: Regular (non-scoped) enum - should NOT warn
enum RegularEnum { X = 1, Y = 2 };

void test_regular_enum() {
  int value = static_cast<int>(RegularEnum::X);
  // No warning expected (only enum class should trigger)
}

// Test case 7: Multiple casts in same function - should warn for each
void test_multiple_casts() {
  int a = static_cast<int>(MyEnum::A);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use 'std::to_underlying' instead of 'static_cast' for 'enum class' [modernize-use-to-underlying]
  // CHECK-FIXES: int a = std::to_underlying(MyEnum::A);
  
  int b = static_cast<int>(MyEnum::B);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use 'std::to_underlying' instead of 'static_cast' for 'enum class' [modernize-use-to-underlying]
  // CHECK-FIXES: int b = std::to_underlying(MyEnum::B);
}

// Test case 8: enum class with underlying type specified
enum class TypedEnum : unsigned int { First = 0, Second = 1 };

void test_typed_enum() {
  unsigned int val = static_cast<unsigned int>(TypedEnum::First);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: use 'std::to_underlying' instead of 'static_cast' for 'enum class' [modernize-use-to-underlying]
  // CHECK-FIXES: unsigned int val = std::to_underlying(TypedEnum::First);
}

// Test case 9: Casting to unsigned - should warn
void test_unsigned_cast() {
  unsigned value = static_cast<unsigned>(MyEnum::A);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use 'std::to_underlying' instead of 'static_cast' for 'enum class' [modernize-use-to-underlying]
  // CHECK-FIXES: unsigned value = std::to_underlying(MyEnum::A);
}

// Test case 10: Nested in function call - should warn
void some_function(int x) {}

void test_nested_call() {
  some_function(static_cast<int>(MyEnum::A));
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use 'std::to_underlying' instead of 'static_cast' for 'enum class' [modernize-use-to-underlying]
  // CHECK-FIXES: some_function(std::to_underlying(MyEnum::A));
}