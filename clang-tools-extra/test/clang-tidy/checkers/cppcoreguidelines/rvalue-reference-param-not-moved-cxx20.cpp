// RUN: %check_clang_tidy -check-suffix=ALLOW-CXX20 -std=c++20-or-later %s cppcoreguidelines-rvalue-reference-param-not-moved %t -- \
// RUN:   -config="{CheckOptions: {cppcoreguidelines-rvalue-reference-param-not-moved.AllowImplicitMove: true}}"
// RUN: %check_clang_tidy -check-suffix=ALLOW-PRE-CXX20 -std=c++11,c++14,c++17 %s cppcoreguidelines-rvalue-reference-param-not-moved %t -- \
// RUN:   -config="{CheckOptions: {cppcoreguidelines-rvalue-reference-param-not-moved.AllowImplicitMove: true}}"
// RUN: %check_clang_tidy -check-suffix=DEFAULT -std=c++11-or-later %s cppcoreguidelines-rvalue-reference-param-not-moved %t

#include <utility>

struct S {
  S();
  S(const S&);
  S(S&&) noexcept;
  S& operator=(const S&);
  S& operator=(S&&) noexcept;
};

int intImplicitReturn(int&& x) {
  // CHECK-MESSAGES-ALLOW-PRE-CXX20: :[[@LINE-1]]:29: warning: rvalue reference parameter 'x' is never moved from inside the function body
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-2]]:29: warning: rvalue reference parameter 'x' is never moved from inside the function body
  return x;
}

S classImplicitReturn(S&& s) {
  // CHECK-MESSAGES-ALLOW-PRE-CXX20: :[[@LINE-1]]:27: warning: rvalue reference parameter 's' is never moved from inside the function body
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-2]]:27: warning: rvalue reference parameter 's' is never moved from inside the function body
  return s;
}

S classImplicitReturnParens(S&& s) {
  // CHECK-MESSAGES-ALLOW-PRE-CXX20: :[[@LINE-1]]:33: warning: rvalue reference parameter 's' is never moved from inside the function body
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-2]]:33: warning: rvalue reference parameter 's' is never moved from inside the function body
  return (s);
}

S explicitMoveReturn(S&& s) {
  return std::move(s);
}

S notMovedOrReturned(S&& s) {
  // CHECK-MESSAGES-ALLOW-CXX20: :[[@LINE-1]]:26: warning: rvalue reference parameter 's' is never moved from inside the function body
  // CHECK-MESSAGES-ALLOW-PRE-CXX20: :[[@LINE-2]]:26: warning: rvalue reference parameter 's' is never moved from inside the function body
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-3]]:26: warning: rvalue reference parameter 's' is never moved from inside the function body
  S copy = s;
  return copy;
}

S SomePathsReturnParam(S&& s, bool cond) {
  // CHECK-MESSAGES-ALLOW-PRE-CXX20: :[[@LINE-1]]:28: warning: rvalue reference parameter 's' is never moved from inside the function body
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-2]]:28: warning: rvalue reference parameter 's' is never moved from inside the function body
  if (cond)
    return s;
  return S();
}

S NoPathReturnsParam(S&& s, bool cond) {
  // CHECK-MESSAGES-ALLOW-CXX20: :[[@LINE-1]]:26: warning: rvalue reference parameter 's' is never moved from inside the function body
  // CHECK-MESSAGES-ALLOW-PRE-CXX20: :[[@LINE-2]]:26: warning: rvalue reference parameter 's' is never moved from inside the function body
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-3]]:26: warning: rvalue reference parameter 's' is never moved from inside the function body
  if (cond)
    return S();
  S copy = s;
  return copy;
}

S TwoParamsBothMoved(S&& a, S&& b, bool cond) {
  // CHECK-MESSAGES-ALLOW-PRE-CXX20: :[[@LINE-1]]:26: warning: rvalue reference parameter 'a' is never moved from inside the function body
  // CHECK-MESSAGES-ALLOW-PRE-CXX20: :[[@LINE-2]]:33: warning: rvalue reference parameter 'b' is never moved from inside the function body
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-3]]:26: warning: rvalue reference parameter 'a' is never moved from inside the function body
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-4]]:33: warning: rvalue reference parameter 'b' is never moved from inside the function body
  if (cond)
    return a;
  return b;
}

S TwoParamsOnlyOneMoved(S&& a, S&& b) {
  // CHECK-MESSAGES-ALLOW-CXX20: :[[@LINE-1]]:36: warning: rvalue reference parameter 'b' is never moved from inside the function body
  // CHECK-MESSAGES-ALLOW-PRE-CXX20: :[[@LINE-2]]:29: warning: rvalue reference parameter 'a' is never moved from inside the function body
  // CHECK-MESSAGES-ALLOW-PRE-CXX20: :[[@LINE-3]]:36: warning: rvalue reference parameter 'b' is never moved from inside the function body
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-4]]:29: warning: rvalue reference parameter 'a' is never moved from inside the function body
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-5]]:36: warning: rvalue reference parameter 'b' is never moved from inside the function body
  (void)b;
  return a;
}

struct A {
  A();
  A(const A&, int);
  A(A&&) noexcept;
};

A explicitCtorWithParamArg(A&& param) {
  // CHECK-MESSAGES-ALLOW-CXX20: :[[@LINE-1]]:32: warning: rvalue reference parameter 'param' is never moved from inside the function body
  // CHECK-MESSAGES-ALLOW-PRE-CXX20: :[[@LINE-2]]:32: warning: rvalue reference parameter 'param' is never moved from inside the function body
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-3]]:32: warning: rvalue reference parameter 'param' is never moved from inside the function body
  return A(param, 10);
}

S explicitCtorCall(S&& s) {
  // CHECK-MESSAGES-ALLOW-CXX20: :[[@LINE-1]]:24: warning: rvalue reference parameter 's' is never moved from inside the function body
  // CHECK-MESSAGES-ALLOW-PRE-CXX20: :[[@LINE-2]]:24: warning: rvalue reference parameter 's' is never moved from inside the function body
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-3]]:24: warning: rvalue reference parameter 's' is never moved from inside the function body
  return S(s);
}
