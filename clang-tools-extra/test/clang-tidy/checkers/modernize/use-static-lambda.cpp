// RUN: %check_clang_tidy -std=c++23-or-later %s modernize-use-static-lambda %t

void basicCases() {
  auto f1 = [](int x) { return x * x; };
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: lambda with empty capture list can be marked 'static' [modernize-use-static-lambda]
  // CHECK-FIXES: auto f1 = [](int x) static { return x * x; };

  auto f2 = [] { return 42; };
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: lambda with empty capture list can be marked 'static' [modernize-use-static-lambda]
  // CHECK-FIXES: auto f2 = [] static { return 42; };

  auto f3 = [](int x) -> int { return x; };
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: lambda with empty capture list can be marked 'static' [modernize-use-static-lambda]
  // CHECK-FIXES: auto f3 = [](int x) static -> int { return x; };

  auto f4 = []() noexcept { return 1; };
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: lambda with empty capture list can be marked 'static' [modernize-use-static-lambda]
  // CHECK-FIXES: auto f4 = []() static noexcept { return 1; };
}

void alreadyStatic() {
  auto f = [](int x) static { return x * x; };
  (void)f;
}

void capturingLambdas() {
  int x = 5;
  auto f1 = [x]() { return x; };
  (void)f1;

  auto f2 = [&x]() { return x; };
  (void)f2;

  // Capture-default counts as a capture even if nothing is actually captured.
  auto f3 = [=]() { return 0; };
  (void)f3;

  auto f4 = [&]() { return 0; };
  (void)f4;
}

void mutableLambda() {
  auto g = []() mutable { return 0; };
  (void)g;
}

#define MAKE_LAMBDA [](int x) { return x; }

void macroLambda() {
  auto f = MAKE_LAMBDA;
  (void)f;
}

void constexprLambda() {
  auto f = []() constexpr { return 42; };
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: lambda with empty capture list can be marked 'static' [modernize-use-static-lambda]
  // CHECK-FIXES: auto f = []() static constexpr { return 42; };
}

void constevalLambda() {
  auto L = []() consteval { return 1; };
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: lambda with empty capture list can be marked 'static' [modernize-use-static-lambda]
  // CHECK-FIXES: auto L = []() static consteval { return 1; };
  (void)L;
}

void combinedSpecifiers() {
  auto L = []() noexcept -> int { return 1; };
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: lambda with empty capture list can be marked 'static' [modernize-use-static-lambda]
  // CHECK-FIXES: auto L = []() static noexcept -> int { return 1; };
  (void)L;
}

void genericLambda() {
  auto L = [](auto X) { return X; };
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: lambda with empty capture list can be marked 'static' [modernize-use-static-lambda]
  // CHECK-FIXES: auto L = [](auto X) static { return X; };
  (void)L;
}

void explicitTemplateParameterList() {
  auto L = []<typename T>(T X) { return X; };
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: lambda with empty capture list can be marked 'static' [modernize-use-static-lambda]
  // CHECK-FIXES: auto L = []<typename T>(T X) static { return X; };
  (void)L;
}

// The check must not produce duplicate diagnostics for template instantiations.
template <typename T>
void templated(T Value) {
  auto L = [](T X) { return X; };
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: lambda with empty capture list can be marked 'static' [modernize-use-static-lambda]
  // CHECK-FIXES: auto L = [](T X) static { return X; };
  (void)L;
}

void instantiateTemplates() {
  templated(1);
  templated(2);
}

// A lambda that is already static via a macro expansion must not be diagnosed.
#define STATIC static

void staticFromMacro() {
  auto L = []() STATIC { return 1; };
  (void)L;
}
