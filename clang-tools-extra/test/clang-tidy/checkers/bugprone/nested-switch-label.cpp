// RUN: %check_clang_tidy %s bugprone-nested-switch-label %t

void directLabels(int Value) {
  switch (Value) {
  case 0:
  case 1:
    break;
  default:
    break;
  }
}

void compoundAfterLabel(int Value) {
  switch (Value) {
  case 0: {
    break;
  }
  default: {
    break;
  }
  }
}

void directLabelInNonCompoundBody(int Value) {
  switch (Value)
  case 0:
    break;
}

void nestedLabelInNonCompoundBody(int Value, bool Condition) {
  switch (Value)
    if (Condition) {
    case 0:
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: switch label is nested inside a compound statement other than the switch body [bugprone-nested-switch-label]
      break;
    }
}

void nestedLabelInLoopBody(int Value, bool Condition) {
  switch (Value)
    while (Condition) {
    case 0:
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: switch label is nested inside a compound statement other than the switch body [bugprone-nested-switch-label]
      break;
    }
}

void nestedCompound(int Value) {
  switch (Value) {
    {
    case 0:
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: switch label is nested inside a compound statement other than the switch body [bugprone-nested-switch-label]
      break;
    default:
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: switch label is nested inside a compound statement other than the switch body [bugprone-nested-switch-label]
      break;
    }
  }
}

void nestedControlFlow(int Value, bool Condition) {
  switch (Value) {
  case 0:
    if (Condition) {
      ++Value;
      break;
    case 1:
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: switch label is nested inside a compound statement other than the switch body [bugprone-nested-switch-label]
      break;
    }
    break;
  }
}

void nestedLabelWithCompound(int Value, bool Condition) {
  switch (Value) {
  case 0:
    if (Condition) {
    case 1: {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: switch label is nested inside a compound statement other than the switch body [bugprone-nested-switch-label]
      break;
    }
    }
    break;
  }
}

void nestedSwitch(int Outer, int Inner) {
  switch (Outer) {
  case 0:
    switch (Inner) {
    case 1:
      break;
    default:
      break;
    }
    break;
  default:
    break;
  }
}

void nestedConsecutiveLabels(int Value) {
  switch (Value) {
    {
    case 0:
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: switch label is nested inside a compound statement other than the switch body [bugprone-nested-switch-label]
    case 1:
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: switch label is nested inside a compound statement other than the switch body [bugprone-nested-switch-label]
      break;
    }
  }
}

template <typename T> void nestedTemplate(T Value) {
  switch (Value) {
    {
    case 0:
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: switch label is nested inside a compound statement other than the switch body [bugprone-nested-switch-label]
      break;
    }
  }
}

void instantiateTemplate() { nestedTemplate(0); }

#define CASE_ONE case 1:

void macroLabel(int Value) {
  switch (Value) {
    {
    CASE_ONE
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: switch label is nested inside a compound statement other than the switch body [bugprone-nested-switch-label]
      break;
    }
  }
}
