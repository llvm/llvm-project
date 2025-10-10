// RUN: %check_clang_tidy %s readability-avoid-default-lambda-capture %t -- -- -Wno-vla-extension

void test_default_captures() {
  int value = 42;
  int another = 10;

  auto lambda1 = [=](int x) { return value + x; };
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: lambda default captures are discouraged; prefer to capture specific variables explicitly [readability-avoid-default-lambda-capture]
  // CHECK-FIXES: auto lambda1 = [value](int x) { return value + x; };

  auto lambda2 = [&](int x) { return value + x; };
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: lambda default captures are discouraged; prefer to capture specific variables explicitly [readability-avoid-default-lambda-capture]
  // CHECK-FIXES: auto lambda2 = [&value](int x) { return value + x; };

  auto lambda3 = [=, &another](int x) { return value + another + x; };
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: lambda default captures are discouraged; prefer to capture specific variables explicitly [readability-avoid-default-lambda-capture]
  // CHECK-FIXES: auto lambda3 = [value, &another](int x) { return value + another + x; };

  auto lambda4 = [&, value](int x) { return value + another + x; };
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: lambda default captures are discouraged; prefer to capture specific variables explicitly [readability-avoid-default-lambda-capture]
  // CHECK-FIXES: auto lambda4 = [&another, value](int x) { return value + another + x; };
}

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
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: lambda default captures are discouraged; prefer to capture specific variables explicitly [readability-avoid-default-lambda-capture]
    // CHECK-FIXES: auto outer = [outer_var, middle_var, inner_var]() {

    auto inner = [&](int x) { return outer_var + middle_var + inner_var + x; };
    // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: lambda default captures are discouraged; prefer to capture specific variables explicitly [readability-avoid-default-lambda-capture]
    // CHECK-FIXES: auto inner = [&outer_var, &middle_var, &inner_var](int x) { return outer_var + middle_var + inner_var + x; };
    
    return inner(10);
  };
}

void test_lambda_returns() {
  int a = 1, b = 2, c = 3;

  auto create_adder = [=](int x) {
    // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: lambda default captures are discouraged; prefer to capture specific variables explicitly [readability-avoid-default-lambda-capture]
    // CHECK-FIXES: auto create_adder = [](int x) {
    return [x](int y) { return x + y; }; // Inner lambda is fine - explicit capture
  };
  
  auto func1 = [&]() { return a; };
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: lambda default captures are discouraged; prefer to capture specific variables explicitly [readability-avoid-default-lambda-capture]
  // CHECK-FIXES: auto func1 = [&a]() { return a; };

  auto func2 = [=]() { return b; };
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: lambda default captures are discouraged; prefer to capture specific variables explicitly [readability-avoid-default-lambda-capture]
  // CHECK-FIXES: auto func2 = [b]() { return b; };
}

class TestClass {
  int member = 42;
  
public:
  void test_member_function_lambdas() {
    int local = 10;
    
    auto lambda1 = [=]() { return member + local; };
    // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: lambda default captures are discouraged; prefer to capture specific variables explicitly [readability-avoid-default-lambda-capture]
    // CHECK-FIXES: auto lambda1 = [this, local]() { return member + local; };

    auto lambda2 = [&]() { return member + local; };
    // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: lambda default captures are discouraged; prefer to capture specific variables explicitly [readability-avoid-default-lambda-capture]
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
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: lambda default captures are discouraged; prefer to capture specific variables explicitly [readability-avoid-default-lambda-capture]
}

void instantiate_templates() {
  test_template_lambdas<int>();
  test_template_lambdas<double>();
}

void test_init_captures() {
  int x = 3;
  int nx = 5;

  int y1 = [&, z = x + 5]() -> int {
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: lambda default captures are discouraged; prefer to capture specific variables explicitly [readability-avoid-default-lambda-capture]
  // CHECK-FIXES: int y1 = [&nx, z = x + 5]() -> int {
    return z * z + nx;
  }();

  int y2 = [=, &ref = x]() {
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: lambda default captures are discouraged; prefer to capture specific variables explicitly [readability-avoid-default-lambda-capture]
  // CHECK-FIXES: int y2 = [nx, &ref = x]() {
    ref += 1;
    return nx - ref;
  }();

  (void)y1;
  (void)y2;
}

void test_vla_no_crash() {
  // VLAs create implicit VLA bound captures that cannot be written explicitly.
  // No warning should be issued.
  int n = 5;
  int vla[n];
  for (int i = 0; i < n; ++i) {
    vla[i] = i * 10;
  }

  auto lambda = [&]() { return vla[0]; };
}
