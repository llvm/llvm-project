// RUN: %check_clang_tidy -std=c++20-or-later %s modernize-use-default-member-init %t

struct PositiveBitField
{
  PositiveBitField() : i(6) {}
  // CHECK-FIXES: PositiveBitField()  {}
  int i : 5;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use default member initializer for 'i' [modernize-use-default-member-init]
  // CHECK-FIXES: int i : 5{6};
};
