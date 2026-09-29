// RUN: %check_clang_tidy -std=c++98 %s bugprone-missing-end-comparison %t

#include <algorithm>

void test_raw_pointers() {
  int arr[] = {1, 2, 3};
  int *begin = arr;
  int *end = arr + 3;

  if (std::find(begin, end, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::find(begin, end, 2) != end)) {}

  if (!std::find(begin, end, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::find(begin, end, 2) == end)) {}

  if (std::lower_bound(begin, end, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::lower_bound(begin, end, 2) != end)) {}

  if (std::search(begin, end, begin, end)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'
  // CHECK-FIXES: if ((std::search(begin, end, begin, end) != end)) {}

  if (std::min_element(begin, end)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'
  // CHECK-FIXES: if ((std::min_element(begin, end) != end)) {}
}

void test_negative() {
  int arr[] = {1, 2, 3};
  int *begin = arr;
  int *end = arr + 3;
  if (std::find(begin, end, 2) != end) {}
}
