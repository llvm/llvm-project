// RUN: %check_clang_tidy -std=c++17-or-later %s bugprone-missing-end-comparison %t

#include <algorithm>

void test_execution_policy() {
  int arr[] = {1, 2, 3};
  int* begin = arr;
  int* end = arr + 3;

  if (std::find(std::execution::seq, begin, end, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::find(std::execution::seq, begin, end, 2) != end)) {}

  if (std::find(std::execution::par, begin, end, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::find(std::execution::par, begin, end, 2) != end)) {}

  auto it = std::find(std::execution::seq, begin, end, 2);
  if (it) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((it != end)) {}

  if (std::lower_bound(std::execution::seq, begin, end, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::lower_bound(std::execution::seq, begin, end, 2) != end)) {}
}

#define FIND_WITH_POLICY(begin, end, val) \
  std::find(std::execution::seq, begin, end, val)

void test_execution_policy_macro() {
  int arr[] = {1, 2, 3};
  int* begin = arr;
  int* end = arr + 3;

  if (FIND_WITH_POLICY(begin, end, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
}

template <typename T>
void test_execution_policy_template_impl(T* begin, T* end, T val) {
  if (std::find(std::execution::seq, begin, end, val)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::find(std::execution::seq, begin, end, val) != end)) {}
}

void test_execution_policy_template() {
  int arr[] = {1, 2, 3};
  test_execution_policy_template_impl(arr, arr + 3, 2);
}
