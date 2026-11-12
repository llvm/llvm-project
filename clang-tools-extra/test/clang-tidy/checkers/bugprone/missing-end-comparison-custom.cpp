// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-missing-end-comparison %t \
// RUN: -config="{CheckOptions: {bugprone-missing-end-comparison.ExtraAlgorithms: '::my_lib::find;::my_lib::find_range'}}"

#include <algorithm>
#include <vector>

namespace my_lib {
  template<typename Iter, typename T>
  Iter find(Iter first, Iter last, const T& value) {
    return first;
  }

  template<typename Range, typename T>
  auto find_range(Range&& range, const T& value)
      -> decltype(range.begin()) {
    return range.begin();
  }
}

void test_custom_algorithms() {
  std::vector<int> v;
  if (my_lib::find(v.begin(), v.end(), 42)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((my_lib::find(v.begin(), v.end(), 42) != v.end())) {}

  if (my_lib::find_range(v, 42)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((my_lib::find_range(v, 42) != std::end(v))) {}
}

void test_still_checks_standard() {
  std::vector<int> v;
  if (std::find(v.begin(), v.end(), 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::find(v.begin(), v.end(), 2) != v.end())) {}

  if (std::min_element(v.begin(), v.end())) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used as 'bool'; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::min_element(v.begin(), v.end()) != v.end())) {}
}
