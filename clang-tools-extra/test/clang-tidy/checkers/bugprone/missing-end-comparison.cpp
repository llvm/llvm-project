// RUN: %check_clang_tidy -std=c++20 %s bugprone-missing-end-comparison %t -- -- -I %S/Inputs/missing-end-comparison

#include "fake_std.h"

struct CustomIterator {
    int* ptr;
    using iterator_category = std::forward_iterator_tag;
    using value_type = int;
    using difference_type = std::ptrdiff_t;
    using pointer = int*;
    using reference = int&;

    int& operator*() const { return *ptr; }
    CustomIterator& operator++() { ++ptr; return *this; }
    bool operator==(const CustomIterator& other) const { return ptr == other.ptr; }
    bool operator!=(const CustomIterator& other) const { return ptr != other.ptr; }

    explicit operator bool() const { return ptr != nullptr; }
};

void test_raw_pointers() {
  int arr[] = {1, 2, 3};
  int* begin = arr;
  int* end = arr + 3;

  if (std::find(begin, end, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::find(begin, end, 2) != end)) {}

  while (std::lower_bound(begin, end, 2)) { break; }
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: while ((std::lower_bound(begin, end, 2) != end)) { break; }

  if (!std::find(begin, end, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::find(begin, end, 2) == end)) {}
}

void test_vector() {
  std::vector<int> v;
  if (std::find(v.begin(), v.end(), 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::find(v.begin(), v.end(), 2) != v.end())) {}

  if (std::min_element(v.begin(), v.end())) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::min_element(v.begin(), v.end()) != v.end())) {}
}

void test_variable_tracking() {
  int arr[] = {1, 2, 3};
  auto it = std::find(arr, arr + 3, 2);
  if (it) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((it != arr + 3)) {}

  if (!it) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((it == arr + 3)) {}
}

void test_ranges() {
  std::vector<int> v;
  if (std::ranges::find(v, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::ranges::find(v, 2) != std::ranges::end(v))) {}

  auto it = std::ranges::find(v, 2);
  if (it) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((it != std::ranges::end(v))) {}

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

void test_ranges_iterator_pair() {
  int arr[] = {1, 2, 3};
  int *begin = arr;
  int *end = arr + 3;
  if (std::ranges::find(begin, end, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::ranges::find(begin, end, 2) != end)) {}
}

void test_side_effects() {
  std::vector<int> get_vec();
  if (std::ranges::find(get_vec(), 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
}

void test_negative() {
  std::vector<int> v;
  if (std::find(v.begin(), v.end(), 2) != v.end()) {}
  if (std::ranges::find(v, 2) == std::ranges::end(v)) {}
  auto it = std::find(v.begin(), v.end(), 2);
}

void test_nested_parens() {
  int arr[] = {1, 2, 3};
  if (!((std::find(arr, arr + 3, 2)))) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((((std::find(arr, arr + 3, 2))) == arr + 3)) {}
}

struct Data { std::vector<int> v; };
void test_member_expr(Data& d) {
  if (std::ranges::find(d.v, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::ranges::find(d.v, 2) != std::ranges::end(d.v))) {}
}

void test_nullptr_comparison() {
  if (std::find((int*)nullptr, (int*)nullptr, 2)) {}
}

void test_double_negation() {
  int arr[] = {1, 2, 3};
  auto it = std::find(arr, arr + 3, 2);
  if (!!it) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if (!(it == arr + 3)) {}
}

void test_custom_iterator() {
  CustomIterator begin{nullptr}, end{nullptr};
  if (std::find(begin, end, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'
  // CHECK-FIXES: if ((std::find(begin, end, 2) != end)) {}
}

void test_search() {
  int arr[] = {1, 2, 3};
  int* begin = arr;
  int* end = arr + 3;

  if (std::search(begin, end, begin, end)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'
  // CHECK-FIXES: if ((std::search(begin, end, begin, end) != end)) {}
}

namespace other {
  bool find(int* b, int* e, int v);
}

void test_other_namespace() {
  int arr[] = {1};
  if (other::find(arr, arr + 1, 1)) {}
}

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

void test_loops() {
  int arr[] = {1, 2, 3};
  int *begin = arr;
  int *end = arr + 3;

  while (std::find(begin, end, 2)) { break; }
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: while ((std::find(begin, end, 2) != end)) { break; }

  do { } while (std::find(begin, end, 2));
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: do { } while ((std::find(begin, end, 2) != end));

  for (auto it = std::find(begin, end, 2); it; ) { break; }
  // CHECK-MESSAGES: :[[@LINE-1]]:44: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: for (auto it = std::find(begin, end, 2); (it != end); ) { break; }

  std::vector<int> v;
  for (auto it = std::ranges::find(v, 2); !it; ) { break; }
  // CHECK-MESSAGES: :[[@LINE-1]]:44: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: for (auto it = std::ranges::find(v, 2); (it == std::ranges::end(v)); ) { break; }
}

void test_condition_variable_suppression() {
  int arr[] = {1, 2, 3};
  if (int* it2 = std::find(arr, arr + 3, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]

  while (int* it3 = std::find(arr, arr + 3, 2)) { break; }
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]

  for (; int* it4 = std::find(arr, arr + 3, 2); ) { break; }
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]

  std::vector<int> v;
  if (auto it5 = std::ranges::find(v, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
}
