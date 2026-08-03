// RUN: %check_clang_tidy %s bugprone-nested-switch-label %t

void direct_labels(int value) {
  switch (value) {
  case 0:
  case 1:
    break;
  default:
    break;
  }
}

void compound_after_label(int value) {
  switch (value) {
  case 0: {
    break;
  }
  default: {
    break;
  }
  }
}

void direct_label_in_non_compound_body(int value) {
  switch (value)
  case 0:
    break;
}

void nested_label_in_non_compound_body(int value, int condition) {
  switch (value)
    if (condition) {
    case 0:
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: switch label is nested inside a compound statement other than the switch body [bugprone-nested-switch-label]
      break;
    }
}

void nested_label_in_loop_body(int value, int condition) {
  switch (value)
    while (condition) {
    case 0:
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: switch label is nested inside a compound statement other than the switch body [bugprone-nested-switch-label]
      break;
    }
}

void nested_compound(int value) {
  switch (value) {
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

void nested_control_flow(int value, int condition) {
  switch (value) {
  case 0:
    if (condition) {
      ++value;
      break;
    case 1:
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: switch label is nested inside a compound statement other than the switch body [bugprone-nested-switch-label]
      break;
    }
    break;
  }
}

void nested_label_with_compound(int value, int condition) {
  switch (value) {
  case 0:
    if (condition) {
    case 1: {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: switch label is nested inside a compound statement other than the switch body [bugprone-nested-switch-label]
      break;
    }
    }
    break;
  }
}

void nested_switch(int outer, int inner) {
  switch (outer) {
  case 0:
    switch (inner) {
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
