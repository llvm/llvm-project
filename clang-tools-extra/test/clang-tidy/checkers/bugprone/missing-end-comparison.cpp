// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-missing-end-comparison %t

#include <algorithm>
#include <vector>

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

void test_nested_parens() {
  int arr[] = {1, 2, 3};
  if (!((std::find(arr, arr + 3, 2)))) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((((std::find(arr, arr + 3, 2))) == arr + 3)) {}
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
}

void test_condition_variable_suppression() {
  int arr[] = {1, 2, 3};
  if (int* it2 = std::find(arr, arr + 3, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]

  while (int* it3 = std::find(arr, arr + 3, 2)) { break; }
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]

  for (; int* it4 = std::find(arr, arr + 3, 2); ) { break; }
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
}

#define FIND_IN(v, val) std::find(v.begin(), v.end(), val)
#define IS_FOUND(it) (it)

void test_macros() {
  std::vector<int> v;
  if (FIND_IN(v, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]

  auto it = FIND_IN(v, 2);
  if (IS_FOUND(it)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
}

template <typename T>
void test_templates_impl(const std::vector<T>& v, T val) {
  if (std::find(v.begin(), v.end(), val)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::find(v.begin(), v.end(), val) != v.end())) {}
}

void test_templates() {
  std::vector<int> v;
  test_templates_impl(v, 2);
}

namespace other {
  bool find(int* b, int* e, int v);
}

void test_negative() {
  std::vector<int> v;
  if (std::find(v.begin(), v.end(), 2) != v.end()) {}

  auto it = std::find(v.begin(), v.end(), 2);

  int arr[] = {1};
  if (other::find(arr, arr + 1, 1)) {}
}

void test_nullptr_comparison() {
  if (std::find((int*)nullptr, (int*)nullptr, 2)) {}
}
