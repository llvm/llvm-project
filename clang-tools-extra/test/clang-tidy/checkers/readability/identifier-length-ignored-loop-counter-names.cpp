// RUN: %check_clang_tidy %s readability-identifier-length %t \
// RUN: -config='{CheckOptions: {readability-identifier-length.IgnoredLoopCounterNames: "^[ijk]$"}}' \
// RUN: -- -fexceptions

void doIt();

void test() {
  for (int i = 0; i < 5; ++i) { doIt(); } // no warning, i allowed
  for (int m = 0; m < 5; ++m)
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: loop variable name 'm' is too short, expected at least 2 characters [readability-identifier-length]
  {
    doIt();
  }
}
