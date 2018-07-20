// We run clang in completion mode to force skipping of function bodies and
// check if the function bodies were skipped by observing the warnings that
// clang produces.
// RUN: not %clang_cc1 -fsyntax-only -code-completion-at=%s:60:1 %s -o - 2>&1 | FileCheck %s
template <class T>
auto not_skipped() {
  int x;
  if (x = 10) {}
  // Check that this function is not skipped.
  // CHECK: 8:9: warning: using the result of an assignment as a condition without parentheses
  return 1;
}

template <class T>
auto lambda_not_skipped = []() {
  int x;
  if (x = 10) {}
  // Check that this function is not skipped.
  // CHECK: 17:9: warning: using the result of an assignment as a condition without parentheses
  return 1;
}

template <class T>
auto skipped() -> T {
  int x;
  if (x = 10) {}
  // Check that this function is skipped.
  // CHECK-NOT: 26:9: warning: using the result of an assignment as a condition without parentheses
  return 1;
};

auto lambda_skipped = []() -> int {
  int x;
  if (x = 10) {}
  // This could potentially be skipped, but it isn't at the moment.
  // CHECK: 34:9: warning: using the result of an assignment as a condition without parentheses
  return 1;
};

template <class T>
decltype(auto)** not_skipped_ptr() {
  int x;
  if (x = 10) {}
  // Check that this function is not skipped.
  // CHECK: 43:9: warning: using the result of an assignment as a condition without parentheses
  return T();
}

template <class T>
decltype(auto) not_skipped_decltypeauto() {
  int x;
  if (x = 10) {}
  // Check that this function is not skipped.
  // CHECK: 52:9: warning: using the result of an assignment as a condition without parentheses
  return 1;
}

int test() {
  int complete_in_this_function;
  // CHECK: COMPLETION: complete_in_this_function
}
