// RUN: %check_clang_tidy %s performance-constexpr-non-static-in-scope %t -- -- -std=c++23

// This should trigger the check (constexpr local, not static).
void f() {
  constexpr int x = 42;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constexpr variable in function scope should be static to ensure static lifetime [performance-constexpr-non-static-in-scope]
  // CHECK-FIXES: static constexpr int x = 42;
}

// This should trigger if WarnInConstexprFuncCpp23 is true and C++23 or newer.
constexpr int g() {
  constexpr int y = 123;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constexpr variable in function scope should be static to ensure static lifetime [performance-constexpr-non-static-in-scope]
  // CHECK-FIXES: static constexpr int y = 123;
  return y;
}

// This should NOT trigger the check (already static).
void h() {
  static constexpr int z = 99;
}

// This should NOT trigger the check (not constexpr).
void i() {
  int w = 100;
}

// This should  NOT trigger the check (not declared inside a function)
namespace ns {
    inline constexpr int MAX_SIZE = 1024;
}