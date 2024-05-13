// RUN: %check_clang_tidy %s bugprone-branch-clone %t -- -- -std=c++17

void handle(int);

template <unsigned Index>
void shouldFail() {
  if constexpr (Index == 0) {
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: repeated branch body in conditional chain [bugprone-branch-clone]
    handle(0);
  } else if constexpr (Index == 1) {
    handle(1);
  } else {
    handle(0);
  }
}

template <unsigned Index>
void shouldPass() {
  if constexpr (Index == 0) {
    handle(0);
  } else if constexpr (Index == 1) {
    handle(1);
  } else {
    handle(2);
  }
}

void shouldFailNonTemplate() {
  constexpr unsigned Index = 1;
  if constexpr (Index == 0) {
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: repeated branch body in conditional chain [bugprone-branch-clone]
    handle(0);
  } else if constexpr (Index == 1) {
    handle(1);
  } else {
    handle(0);
  }
}

void shouldPassNonTemplate() {
  constexpr unsigned Index = 1;
  if constexpr (Index == 0) {
    handle(0);
  } else if constexpr (Index == 1) {
    handle(1);
  } else {
    handle(2);
  }
}

void run() {
    shouldFail<0>();
    shouldFail<1>();
    shouldFail<2>();
    shouldPass<0>();
    shouldPass<1>();
    shouldPass<2>();
}
