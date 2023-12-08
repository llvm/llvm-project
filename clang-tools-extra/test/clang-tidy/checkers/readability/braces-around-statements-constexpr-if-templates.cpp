// RUN: %check_clang_tidy %s readability-braces-around-statements %t -- -- -std=c++17

void handle(bool);

template <bool branch>
void shouldFail() {
  if constexpr (branch)
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: statement should be inside braces [readability-braces-around-statements]
    handle(true);
  else
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: statement should be inside braces [readability-braces-around-statements]
    handle(false);
}

template <bool branch>
void shouldPass() {
  if constexpr (branch) {
    handle(true);
  } else {
    handle(false);
  }
}

void shouldFailNonTemplate() {
  constexpr bool branch = false;
  if constexpr (branch)
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: statement should be inside braces [readability-braces-around-statements]
    handle(true);
  else
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: statement should be inside braces [readability-braces-around-statements]
    handle(false);
}

void shouldPass() {
  constexpr bool branch = false;
  if constexpr (branch) {
    handle(true);
  } else {
    handle(false);
  }
}

void run() {
    shouldFail<true>();
    shouldFail<false>();
    shouldPass<true>();
    shouldPass<false>();
}
