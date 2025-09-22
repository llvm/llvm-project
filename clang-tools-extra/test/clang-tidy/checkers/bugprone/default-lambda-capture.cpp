// RUN: %check_clang_tidy %s bugprone-default-lambda-capture %t

void test_default_captures() {
  int value = 42;
  int another = 10;

  auto lambda1 = [=](int x) { return value + x; };
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: lambda default captures are discouraged; prefer to capture specific variables explicitly [bugprone-default-lambda-capture]

  auto lambda2 = [&](int x) { return value + x; };
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: lambda default captures are discouraged; prefer to capture specific variables explicitly [bugprone-default-lambda-capture]

  auto lambda3 = [=, &another](int x) { return value + another + x; };
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: lambda default captures are discouraged; prefer to capture specific variables explicitly [bugprone-default-lambda-capture]

  auto lambda4 = [&, value](int x) { return value + another + x; };
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: lambda default captures are discouraged; prefer to capture specific variables explicitly [bugprone-default-lambda-capture]
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
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: lambda default captures are discouraged; prefer to capture specific variables explicitly [bugprone-default-lambda-capture]
    
    auto inner = [&](int x) { return outer_var + middle_var + inner_var + x; };
    // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: lambda default captures are discouraged; prefer to capture specific variables explicitly [bugprone-default-lambda-capture]
    
    return inner(10);
  };
}

void test_lambda_returns() {
  int a = 1, b = 2, c = 3;

  auto create_adder = [=](int x) {
    // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: lambda default captures are discouraged; prefer to capture specific variables explicitly [bugprone-default-lambda-capture]
    return [x](int y) { return x + y; }; // Inner lambda is fine - explicit capture
  };
  
  auto func1 = [&]() { return a; };
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: lambda default captures are discouraged; prefer to capture specific variables explicitly [bugprone-default-lambda-capture]
  
  auto func2 = [=]() { return b; };
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: lambda default captures are discouraged; prefer to capture specific variables explicitly [bugprone-default-lambda-capture]
}

class TestClass {
  int member = 42;
  
public:
  void test_member_function_lambdas() {
    int local = 10;
    
    auto lambda1 = [=]() { return member + local; };
    // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: lambda default captures are discouraged; prefer to capture specific variables explicitly [bugprone-default-lambda-capture]
    
    auto lambda2 = [&]() { return member + local; };
    // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: lambda default captures are discouraged; prefer to capture specific variables explicitly [bugprone-default-lambda-capture]
    
    auto lambda3 = [this, local]() { return member + local; };
    auto lambda4 = [this, &local]() { return member + local; };
  }
};

template<typename T>
void test_template_lambdas() {
  T value{};
  
  auto lambda = [=](T x) { return value + x; };
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: lambda default captures are discouraged; prefer to capture specific variables explicitly [bugprone-default-lambda-capture]
}

void instantiate_templates() {
  test_template_lambdas<int>();
  test_template_lambdas<double>();
}
