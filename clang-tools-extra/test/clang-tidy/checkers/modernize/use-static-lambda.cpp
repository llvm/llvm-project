// RUN: %check_clang_tidy -std=c++23-or-later %s modernize-use-static-lambda %t

// Basic cases

void basicCases() {
  // Lambda with explicit params, no capture.
  auto f1 = [](int x) { return x * x; };
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: lambda with empty capture list can be marked 'static' [modernize-use-static-lambda]
  // CHECK-FIXES: auto f1 = [](int x) static { return x * x; };

  // Lambda with no params, no capture.
  auto f2 = [] { return 42; };
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: lambda with empty capture list can be marked 'static' [modernize-use-static-lambda]
  // CHECK-FIXES: auto f2 = []() static { return 42; };

  // Lambda with trailing return type.
  auto f3 = [](int x) -> int { return x; };
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: lambda with empty capture list can be marked 'static' [modernize-use-static-lambda]
  // CHECK-FIXES: auto f3 = [](int x) static -> int { return x; };

  // Lambda with noexcept.
  auto f4 = []() noexcept { return 1; };
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: lambda with empty capture list can be marked 'static' [modernize-use-static-lambda]
  // CHECK-FIXES: auto f4 = []() static noexcept { return 1; };
}

// Already static: no warning

void alreadyStatic() {
  auto f = [](int x) static { return x * x; };
  (void)f;
}

// Capturing lambdas: no warning

void capturingLambdas() {
  int x = 5;

  // Explicit capture by value.
  auto f1 = [x]() { return x; };
  (void)f1;

  // Explicit capture by reference.
  auto f2 = [&x]() { return x; };
  (void)f2;

  // Capture-default by value (even if nothing actually captured).
  auto f3 = [=]() { return 0; };
  (void)f3;

  // Capture-default by reference (even if nothing actually captured).
  auto f4 = [&]() { return 0; };
  (void)f4;
}

// Mutable lambda: no warning (static and mutable are mutually exclusive)

void mutableLambda() {
  // Mutable with no capture is still excluded.
  auto g = []() mutable { return 0; };
  (void)g;
}

// Macro: no warning

#define MAKE_LAMBDA [](int x) { return x; }

void macroLambda() {
  auto f = MAKE_LAMBDA;
  (void)f;
}

// constexpr lambda: warning (static + constexpr is valid in C++23)

void constexprLambda() {
  auto f = []() constexpr { return 42; };
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: lambda with empty capture list can be marked 'static' [modernize-use-static-lambda]
  // CHECK-FIXES: auto f = []() static constexpr { return 42; };
}
