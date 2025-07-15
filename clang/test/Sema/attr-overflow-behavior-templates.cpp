// RUN: %clang_cc1 %s -foverflow-behavior-types -verify -fsyntax-only
// expected-no-diagnostics

#define __wrap __attribute__((overflow_behavior(wrap)))
#define __no_wrap __attribute__((overflow_behavior(no_wrap)))

template<typename T>
constexpr int template_overload_test(T) {
  return 10;
}

constexpr int template_overload_test(int) {
  return 20;
}

constexpr int template_overload_test(__wrap int) {
  return 30;
}

constexpr int template_overload_test(__no_wrap int) {
  return 40;
}

void test_template_overload_resolution() {
  static_assert(template_overload_test(42) == 20, "int should pick int overload");
  static_assert(template_overload_test((__wrap int)42) == 30, "__wrap int should pick __wrap int overload");
  static_assert(template_overload_test((__no_wrap int)42) == 40, "__no_wrap int should pick __no_wrap int overload");
}

template<typename T>
struct MultiSpecTester {
  static constexpr int value = 0;
};

template<>
struct MultiSpecTester<int> {
  static constexpr int value = 1;
};

void test_choosing_generic_when_only_underlying_present() {
  static_assert(MultiSpecTester<int>::value == 1, "int should match int specialization");
  // OBTs don't choose template specialization based on underlying type, they should go with the generic
  static_assert(MultiSpecTester<__wrap int>::value == 0, "__wrap int should match generic when there isn't a __wrap specialization");
  static_assert(MultiSpecTester<__no_wrap int>::value == 0, "__no_wrap int should match generic when there isn't a __no_wrap specialization");
}

template<typename T = int>
constexpr int only_int_template(int value) {
  return value + 100;
}

void test_template_conversion_fallback() {
  static_assert(only_int_template<int>(42) == 142, "int direct match should work");
  static_assert(only_int_template<int>((__wrap int)42) == 142, "__wrap int implicit conversion should work");
  static_assert(only_int_template<int>((__no_wrap int)42) == 142, "__no_wrap int implicit conversion should work");
}

void simple_overload_test(int);
void simple_overload_test(__wrap int);

template<typename T>
void simple_overload_test(T) {}

void test_function_vs_template_overload() {
  int regular = 42;
  __wrap int wrapped = 42;
  __no_wrap int no_wrap = 42;

  simple_overload_test(regular);
  simple_overload_test(wrapped);
  simple_overload_test(no_wrap);
}
