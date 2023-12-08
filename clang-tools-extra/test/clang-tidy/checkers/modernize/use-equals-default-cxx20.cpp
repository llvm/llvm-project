// RUN: %check_clang_tidy -std=c++20 %s modernize-use-equals-default %t -- -- -fno-delayed-template-parsing -fexceptions

// Private constructor/destructor.
class Priv {
  Priv() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: Priv() = default;
  ~Priv() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= default'
  // CHECK-FIXES: ~Priv() = default;
};

class PrivOutOfLine {
  PrivOutOfLine();
};

PrivOutOfLine::PrivOutOfLine() {}
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use '= default'
// CHECK-FIXES: PrivOutOfLine::PrivOutOfLine() = default;
