// RUN: %check_clang_tidy %s bugprone-branch-clone %t -- -- -std=c++17

void handle(int);

void testSwitchFallthroughAttribute(int value) {
  switch(value) {
    case 1: [[fallthrough]];
    case 2: [[fallthrough]];
    case 3:
      handle(value);
      break;
    default:
      break;
  }
}

void testSwitchFallthroughAttributeAndBraces(int value) {
  switch(value) {
    case 1: { [[fallthrough]]; }
    case 2: { [[fallthrough]]; }
    case 3: {
      handle(value);
      break;
    }
    default: {
      break;
    }
  }
}

void testSwitchWithFallthroughAttributeAndCode(int value) {
  switch(value) {
    case 1: value += 1; [[fallthrough]];
    case 2: value += 1; [[fallthrough]];
    case 3:
      handle(value);
      break;
    default:
      break;
  }
}

void testSwitchWithFallthroughAndCode(int value) {
  switch(value) {
    // CHECK-MESSAGES: :[[@LINE+1]]:5: warning: switch has 2 consecutive identical branches [bugprone-branch-clone]
    case 1: value += 1;
    case 2: value += 1;
    // CHECK-MESSAGES: :[[@LINE-1]]:23: note: last of these clones ends here
    case 3:
      handle(value);
      break;
    default:
      break;
  }
}

void testSwitchFallthroughAttributeIntoDefault(int value) {
  switch(value) {
    case 1: [[fallthrough]];
    case 2: [[fallthrough]];
    default:
      handle(value);
      break;
  }
}
