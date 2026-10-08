// RUN: %check_clang_tidy %s readability-identifier-length %t \
// RUN: -config='{CheckOptions: {readability-identifier-length.MinimumLoopCounterNameLength: 4}}' \
// RUN: -- -fexceptions

void doIt();

void test() {
  for (int idx = 0; idx < 5; ++idx)
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: loop variable name 'idx' is too short, expected at least 4 characters [readability-identifier-length]
  {
    doIt();
  }
  for (int index = 0; index < 5; ++index) { doIt(); } // 5 chars, ok
}
