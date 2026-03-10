// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.core.StoreToImmutable -std=c++11 -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.core.StoreToImmutable -std=c++14 -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.core.StoreToImmutable -std=c++17 -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.core.StoreToImmutable -std=c++20 -verify %s

// expected-no-diagnostics

// In C++14 and before, when initializing a lambda, the statement given in the checkBind callback is not the whole DeclExpr, but the CXXConstructExpr of the lambda object.
// FIXME: Once the API of checkBind provides more information about the statement, the checker should be simplified, and this test case will no longer be a cornercase in the checker.

void test_const_lambda_initialization_pre_cpp17() {
  const auto lambda = [](){}; // No warning expected
}
