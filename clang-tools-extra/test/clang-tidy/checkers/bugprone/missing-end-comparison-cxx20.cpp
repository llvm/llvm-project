// RUN: %check_clang_tidy -std=c++20-or-later %s bugprone-missing-end-comparison %t

#include <algorithm>
#include <vector>

void test_ranges_range_overload() {
  std::vector<int> v;
  if (std::ranges::find(v, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::ranges::find(v, 2) != std::ranges::end(v))) {}

  auto it = std::ranges::find(v, 2);
  if (it) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((it != std::ranges::end(v))) {}

  if (!std::ranges::find(v, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::ranges::find(v, 2) == std::ranges::end(v))) {}
}

void test_ranges_iterator_pair() {
  int arr[] = {1, 2, 3};
  int *begin = arr;
  int *end = arr + 3;

  if (std::ranges::find(begin, end, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::ranges::find(begin, end, 2) != end)) {}
}

void test_ranges_multi_range() {
  std::vector<int> v1, v2;
  int arr[] = {1, 2, 3};
  int *begin = arr;
  int *end = arr + 3;

  if (std::ranges::find_first_of(v1, v2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::ranges::find_first_of(v1, v2) != std::ranges::end(v1))) {}

  if (std::ranges::find_first_of(begin, end, begin, end)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::ranges::find_first_of(begin, end, begin, end) != end)) {}

  if (std::ranges::adjacent_find(v1)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::ranges::adjacent_find(v1) != std::ranges::end(v1))) {}

  if (std::ranges::adjacent_find(begin, end)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::ranges::adjacent_find(begin, end) != end)) {}

  if (std::ranges::is_sorted_until(v1)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::ranges::is_sorted_until(v1) != std::ranges::end(v1))) {}

  if (std::ranges::is_sorted_until(begin, end)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::ranges::is_sorted_until(begin, end) != end)) {}
}

void test_ranges_loops() {
  std::vector<int> v;
  int arr[] = {1, 2, 3};
  int *begin = arr;
  int *end = arr + 3;

  for (auto it = std::ranges::find(v, 2); !it; ) { break; }
  // CHECK-MESSAGES: :[[@LINE-1]]:44: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: for (auto it = std::ranges::find(v, 2); (it == std::ranges::end(v)); ) { break; }
}

struct Data { std::vector<int> v; };

void test_member_expr(Data& d) {
  if (std::ranges::find(d.v, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::ranges::find(d.v, 2) != std::ranges::end(d.v))) {}
}

void test_side_effects() {
  std::vector<int> get_vec();
  if (std::ranges::find(get_vec(), 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
}

void test_ranges_condition_variable_suppression() {
  std::vector<int> v;
  if (auto it = std::ranges::find(v, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
}

template <typename T>
void test_ranges_templates_impl(const std::vector<T>& v, T val) {
  auto it = std::ranges::find(v, val);
  if (it) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((it != std::ranges::end(v))) {}
}

void test_ranges_templates() {
  std::vector<int> v;
  test_ranges_templates_impl(v, 2);
}

#define RANGES_FIND_IN(v, val) std::ranges::find(v, val)
#define IS_FOUND(it) (it)

void test_ranges_macros() {
  std::vector<int> v;
  if (RANGES_FIND_IN(v, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]

  auto it = RANGES_FIND_IN(v, 2);
  if (IS_FOUND(it)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
}

void test_ranges_negative() {
  std::vector<int> v;
  if (std::ranges::find(v, 2) == std::ranges::end(v)) {}
}
