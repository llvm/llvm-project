// RUN: %check_clang_tidy -std=c++98 %s modernize-use-equals-default %t

struct S {
  S() {}
  // CHECK-FIXES: S() {}
  ~S() {}
  // CHECK-FIXES: ~S() {}
};
