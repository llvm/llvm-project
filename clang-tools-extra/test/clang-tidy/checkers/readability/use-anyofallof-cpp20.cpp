// RUN: %check_clang_tidy -std=c++20-or-later %s readability-use-anyofallof %t

#include <vector>
#include <initializer_list>

bool good_any_of() {
  int v[] = {1, 2, 3};
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: replace loop by 'std::ranges::any_of()'
  for (int i : v)
    if (i)
      return true;
  return false;
}

bool good_all_of() {
  int v[] = {1, 2, 3};
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: replace loop by 'std::ranges::all_of()'
  for (int i : v)
    if (i)
      return false;
  return true;
}

std::vector<int> get_dummy_vec();

bool good_any_of_temporary_vector() {
  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: replace loop by 'std::ranges::any_of()'
  // CHECK-MESSAGES: :[[@LINE+1]]:16: note: reusing the temporary range directly in the replacement may be unsafe; consider materializing it in a local variable first, or use 'std::ranges' algorithms which handle temporary ranges safely
  for (int i : get_dummy_vec())
    if (i)
      return true;
  return false;
}

bool good_all_of_temporary_vector() {
  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: replace loop by 'std::ranges::all_of()'
  // CHECK-MESSAGES: :[[@LINE+1]]:16: note: reusing the temporary range directly in the replacement may be unsafe; consider materializing it in a local variable first, or use 'std::ranges' algorithms which handle temporary ranges safely
  for (int i : get_dummy_vec())
    if (i)
      return false;
  return true;
}

bool any_of_initializer_list(int a, int b, int c) {
  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: replace loop by 'std::ranges::any_of()'
  // CHECK-MESSAGES: :[[@LINE+1]]:23: note: reusing the temporary range directly in the replacement may be unsafe; consider materializing it in a local variable first, or use 'std::ranges' algorithms which handle temporary ranges safely
  for (const auto i : {a, b, c})
    if (i == 0)
      return true;
  return false;
}

bool all_of_initializer_list(int a, int b, int c) {
  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: replace loop by 'std::ranges::all_of()'
  // CHECK-MESSAGES: :[[@LINE+1]]:23: note: reusing the temporary range directly in the replacement may be unsafe; consider materializing it in a local variable first, or use 'std::ranges' algorithms which handle temporary ranges safely
  for (const auto i : {a, b, c})
    if (i == 0)
      return false;
  return true;
}

bool good_any_of_explicit_initializer_list(int a, int b, int c) {
  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: replace loop by 'std::ranges::any_of()'
  // CHECK-MESSAGES: :[[@LINE+1]]:23: note: reusing the temporary range directly in the replacement may be unsafe; consider materializing it in a local variable first, or use 'std::ranges' algorithms which handle temporary ranges safely
  for (const auto i : std::initializer_list<int>{a, b, c})
    if (i == 0)
      return true;
  return false;
}

bool good_all_of_explicit_initializer_list(int a, int b, int c) {
  // CHECK-MESSAGES: :[[@LINE+2]]:3: warning: replace loop by 'std::ranges::all_of()'
  // CHECK-MESSAGES: :[[@LINE+1]]:23: note: reusing the temporary range directly in the replacement may be unsafe; consider materializing it in a local variable first, or use 'std::ranges' algorithms which handle temporary ranges safely
  for (const auto i : std::initializer_list<int>{a, b, c})
    if (i == 0)
      return false;
  return true;
}
