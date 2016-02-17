// RUN: %check_clang_tidy %s readability-braces-around-statements %t

int test_failure() {
  if (std::rand()) {
  // CHECK-MESSAGES: :[[@LINE-1]]:7: error: use of undeclared identifier 'std'
  }
}
