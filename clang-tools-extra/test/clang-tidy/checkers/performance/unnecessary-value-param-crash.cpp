// RUN: %check_clang_tidy  -std=c++14-or-later %s performance-unnecessary-value-param %t

// The test case used to crash clang-tidy.
// https://github.com/llvm/llvm-project/issues/108963

struct A
{
  template<typename T> A(T&&) {}
};

struct B
{
  ~B();
};

struct C
{
  A a;
  C(B, int i) : a(i) {}
  // CHECK-MESSAGES: [[@LINE-1]]:6: warning: the parameter #1 is copied for each invocation but only used as a const reference; consider making it a const reference
};

C c(B(), 0);
