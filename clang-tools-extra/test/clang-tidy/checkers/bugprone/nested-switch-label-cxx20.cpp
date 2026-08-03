// RUN: %check_clang_tidy -std=c++20-or-later %s bugprone-nested-switch-label %t

void directLabelInAttributedCompoundBody(int Value) {
  switch (Value) [[likely]] {
  case 0:
    break;
  }
}

void attributedCompoundBody(int Value) {
  switch (Value) [[likely]] {
    {
    case 0:
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: switch label is nested inside a compound statement other than the switch body [bugprone-nested-switch-label]
      break;
    }
  }
}

void attributedBoundaries(int Value, bool Condition) {
  switch (Value) {
  [[likely]] case 0:
    break;
  case 1:
    if (Condition) [[likely]] {
    case 2:
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: switch label is nested inside a compound statement other than the switch body [bugprone-nested-switch-label]
      break;
    }
  }
}
