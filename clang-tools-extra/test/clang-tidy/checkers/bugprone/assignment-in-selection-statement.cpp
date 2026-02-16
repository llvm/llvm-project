// RUN: %check_clang_tidy %s -std=c++17 bugprone-assignment-in-selection-statement %t

struct S {
  int A = 1;
  S &operator=(const S &s) { A = s.A; return *this; }
  operator bool() { return A == 1; }
};

void test(S a) {
  S x;
  int y, z;

  if (x = a) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: assignment within condition of 'if' statement may indicate programmer error
  if (int x = y; x = 0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: assignment within condition of 'if' statement may indicate programmer error

  if ((x = a)) {}
  if ((x = a).A > 1) {}
  if (static_cast<bool>(x = a)) {}
  if (int x = y; x > 0) {}
  if ([&y](int i) { return y = i; }(z = 2)) {}
}

template<typename... Args>
void test_fold(int x, Args... args) {
  if ((... || (args = x))) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: assignment within operand of a logical operator may indicate programmer error
  if ((... = args) && x > 2) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: assignment within operand of a logical operator may indicate programmer error
}

template<typename... Args>
void test_fold_arg(Args... args) {
  if ((... && args)) {}
}

void test1(int x1, int x2, int y1, int y2) {
  test_fold(1, 2, x1);
  test_fold_arg(x1 == y1, x2 = y2);
}
