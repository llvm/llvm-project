// RUN: %check_clang_tidy %s bugprone-too-small-loop-variable %t -- -- --target=x86_64-linux

// MagnitudeBitsUpperLimit = 16 (default value)

unsigned long size() { return 294967296l; }

void voidFilteredOutForLoop1() {
  for (long i = 0; i < size(); ++i) {
    // no warning
  }
}

void voidCaughtForLoop1() {
  for (int i = 0; i < size(); ++i) {
    // no warning
  }
}

void voidCaughtForLoop2() {
  for (short i = 0; i < size(); ++i) {
    // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: loop variable has narrower type 'short' than iteration's upper bound 'unsigned long' [bugprone-too-small-loop-variable]
  }
}
