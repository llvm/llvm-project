// RUN: %check_clang_tidy -std=c++11,c++14,c++17 -check-suffixes=,DEFAULT %s readability-avoid-default-lambda-capture %t -- -- -Wno-vla-extension
// RUN: %check_clang_tidy -std=c++20-or-later -check-suffixes=,DEFAULT,20 %s readability-avoid-default-lambda-capture %t -- -- -Wno-vla-extension
// RUN: %check_clang_tidy -std=c++11,c++14,c++17 -check-suffix= %s readability-avoid-default-lambda-capture %t -- -config="{CheckOptions: {readability-avoid-default-lambda-capture.IgnoreInSTL: true}}" -- -Wno-vla-extension
// RUN: %check_clang_tidy -std=c++20-or-later -check-suffixes=,20 %s readability-avoid-default-lambda-capture %t -- -config="{CheckOptions: {readability-avoid-default-lambda-capture.IgnoreInSTL: true}}" -- -Wno-vla-extension

void test_default_captures() {
  int value = 42;
  int another = 10;

  auto lambda1 = [=](int x) { return value + x; };
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]
  // CHECK-FIXES: auto lambda1 = [value](int x) { return value + x; };

  auto lambda2 = [&](int x) { return value + x; };
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]
  // CHECK-FIXES: auto lambda2 = [&value](int x) { return value + x; };

  auto lambda3 = [=, &another](int x) { return value + another + x; };
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]
  // CHECK-FIXES: auto lambda3 = [value, &another](int x) { return value + another + x; };

  auto lambda4 = [&, value](int x) { return value + another + x; };
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]
  // CHECK-FIXES: auto lambda4 = [&another, value](int x) { return value + another + x; };
}

#if __cplusplus >= 202002L
template<typename... Args>
void test_pack_expansion_captures(Args... args) {
  int local = 5;

  auto lambda1 = [=]() { return (args + ...); };
  // CHECK-MESSAGES-20: :[[@LINE-1]]:19: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]

  auto lambda2 = [&]() { return (args + ...); };
  // CHECK-MESSAGES-20: :[[@LINE-1]]:19: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]

  auto lambda3 = [=]() { return (args + ...) + local; };
  // CHECK-MESSAGES-20: :[[@LINE-1]]:19: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]

  auto lambda4 = [&]() { return (args + ...) + local; };
  // CHECK-MESSAGES-20: :[[@LINE-1]]:19: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]

  auto lambda5 = [=, ...copied = args]() { return (copied + ...); };
  // CHECK-MESSAGES-20: :[[@LINE-1]]:19: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]

  auto lambda6 = [&, ...refs = args]() { return (refs + ...); };
  // CHECK-MESSAGES-20: :[[@LINE-1]]:19: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]
}

void instantiate_pack_expansion_tests() {
  test_pack_expansion_captures(1, 2, 3);
  test_pack_expansion_captures(1.0, 2.0, 3.0);
}
#endif

void test_acceptable_captures() {
  int value = 42;
  int another = 10;

  auto lambda1 = [value](int x) { return value + x; };
  auto lambda2 = [&value](int x) { return value + x; };
  auto lambda3 = [value, another](int x) { return value + another + x; };
  auto lambda4 = [&value, &another](int x) { return value + another + x; };

  auto lambda5 = [](int x, int y) { return x + y; };

  struct S {
    int member = 5;
    void foo() {
      auto lambda = [this]() { return member; };
    }
  };
}

void test_nested_lambdas() {
  int outer_var = 1;
  int middle_var = 2;
  int inner_var = 3;

  auto outer = [=]() {
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]
    // CHECK-FIXES: auto outer = [outer_var, middle_var, inner_var]() {

    auto inner = [&](int x) { return outer_var + middle_var + inner_var + x; };
    // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]
    // CHECK-FIXES: auto inner = [&outer_var, &middle_var, &inner_var](int x) { return outer_var + middle_var + inner_var + x; };

    return inner(10);
  };
}

void test_lambda_returns() {
  int a = 1, b = 2, c = 3;

  auto create_adder = [=](int x) {
    // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]
    // CHECK-FIXES: auto create_adder = [](int x) {
    return [x](int y) { return x + y; }; // Inner lambda is fine - explicit capture
  };

  auto func1 = [&]() { return a; };
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]
  // CHECK-FIXES: auto func1 = [&a]() { return a; };

  auto func2 = [=]() { return b; };
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]
  // CHECK-FIXES: auto func2 = [b]() { return b; };
}

class TestClass {
  int member = 42;

public:
  void test_member_function_lambdas() {
    int local = 10;

    auto lambda1 = [=]() { return member + local; };
    // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]
    // CHECK-FIXES: auto lambda1 = [this, local]() { return member + local; };

    auto lambda2 = [&]() { return member + local; };
    // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]
    // CHECK-FIXES: auto lambda2 = [this, &local]() { return member + local; };

    auto lambda3 = [this, local]() { return member + local; };
    auto lambda4 = [this, &local]() { return member + local; };
  }
};

// Lambda captures dependent on a template parameter don't have a fix it
template<typename T>
void test_template_lambdas() {
  T value{};

  auto lambda = [=](T x) { return value + x; };
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]
}

void instantiate_templates() {
  test_template_lambdas<int>();
  test_template_lambdas<double>();
}

void test_init_captures() {
  int x = 3;
  int nx = 5;

  int y1 = [&, z = x + 5]() -> int {
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]
  // CHECK-FIXES: int y1 = [&nx, z = x + 5]() -> int {
    return z * z + nx;
  }();

  int y2 = [=, &ref = x]() {
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]
  // CHECK-FIXES: int y2 = [nx, &ref = x]() {
    ref += 1;
    return nx - ref;
  }();

  int y3 = [=, &ref = x, z = x + 5]() {
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]
  // CHECK-FIXES: int y3 = [nx, &ref = x, z = x + 5]() {
    ref += 2;
    return nx + z - ref;
  }();

  (void)y1;
  (void)y2;
  (void)y3;
}

void test_vla() {
  // VLAs create implicit VLA bound captures that cannot be written explicitly.
  // No warning should be issued.
  int n = 5;
  int vla[n];

  auto lambda = [&]() { return vla[0]; };
}

// Mock the STL
namespace std {
  template<class InputIt, class UnaryFunction>
  UnaryFunction for_each(InputIt first, InputIt last, UnaryFunction f) {
    return f;
  }
  
  template<class InputIt, class T>
  T accumulate(InputIt first, InputIt last, T init) {
    return init;
  }

  template<class InputIt, class T, class BinaryOperation>
  T accumulate(InputIt first, InputIt last, T init, BinaryOperation op) {
    return init;
  }

  // Mock std::ranges with member function-like callables
  namespace ranges {
    struct {
      template<class Range, class Pred>
      bool operator()(Range&& r, Pred pred) const { return true; }
    } all_of;

    struct {
      template<class Range, class Pred>
      bool operator()(Range&& r, Pred pred) const { return true; }
    } any_of;

    struct {
      template<class Range, class Pred>
      bool operator()(Range&& r, Pred pred) const { return true; }
    } none_of;
  }
}

void test_stl_captures() {
  int x = 10;
  int arr[] = {1, 2, 3};

  // Test free functions in std namespace - should be ignored with IgnoreInSTL
  std::for_each(arr, arr + 3, [=](int i) { });
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:32: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]
  // CHECK-FIXES-DEFAULT: std::for_each(arr, arr + 3, [](int i) { });

  std::accumulate(arr, arr + 3, 0, [&](int a, int b) { return a + b + x; });
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:37: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]
  // CHECK-FIXES-DEFAULT: std::accumulate(arr, arr + 3, 0, [&x](int a, int b) { return a + b + x; });
}

// Test that IgnoreInSTL option also applies to niebloids (algorithmic function objects)
void test_stl_ranges_captures() {
  int x = 10;
  int arr[] = {1, 2, 3};

  std::ranges::all_of(arr, [=](int i) { return i > x; });
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:28: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]
  // CHECK-FIXES-DEFAULT: std::ranges::all_of(arr, [x](int i) { return i > x; });

  std::ranges::any_of(arr, [&](int i) { return i > x; });
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:28: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]
  // CHECK-FIXES-DEFAULT: std::ranges::any_of(arr, [&x](int i) { return i > x; });

  std::ranges::none_of(arr, [=](int i) { return i < 0; });
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:29: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]
  // CHECK-FIXES-DEFAULT: std::ranges::none_of(arr, [](int i) { return i < 0; });
}

// Test that nested lambdas inside STL function calls are also ignored
// when IgnoreInSTL is enabled (due to hasAncestor matching).
// This is intentional: the nested lambda is still immediately executed
// within the STL algorithm and won't outlive the call.
void test_stl_nested_lambda() {
  int x = 10;
  int arr[] = {1, 2, 3};

  std::for_each(arr, arr + 3, [](int i) {
    // Outer lambda has no default capture - OK

    // Nested lambda with default capture inside STL call.
    // With IgnoreInSTL=true, this also doesn't warn because
    // it has an ancestor in std namespace.
    auto nested = [&]() { return i; };
    // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:19: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]
    // CHECK-FIXES-DEFAULT: auto nested = [&i]() { return i; };
    (void)nested;
  });

  std::ranges::all_of(arr, [](int i) {
    auto nested = [=]() { return i * 2; };
    // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:19: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]
    // CHECK-FIXES-DEFAULT: auto nested = [i]() { return i * 2; };
    (void)nested;
    return true;
  });
}

// Lambdas in macros should warn but not provide fix-its
#define MACRO_WITH_LAMBDA(x) [=]() { return x; }
#define BY_VALUE =
#define BY_REF &

void test_macros() {
  int value = 42;

  // Lambda defined in macro body - warn but no fix-it
  auto lambda1 = MACRO_WITH_LAMBDA(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]

  // Default capture from macro - warn but no fix-it
  auto lambda2 = [BY_VALUE]() { return value; };
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]

  auto lambda3 = [BY_REF]() { return value; };
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: lambda uses default capture mode; explicitly capture variables instead [readability-avoid-default-lambda-capture]
}
