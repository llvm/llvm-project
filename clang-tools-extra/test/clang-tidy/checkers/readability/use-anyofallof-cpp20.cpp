// RUN: %check_clang_tidy -std=c++20-or-later %s readability-use-anyofallof %t

bool good_any_of() {
  int v[] = {1, 2, 3};
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: replace loop by 'std::ranges::any_of()'
  for (int i : v)
    if (i)
      return true;
  return false;
}

bool good_none_of() {
  int v[] = {1, 2, 3};
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: replace loop by 'std::ranges::none_of()'
  for (int i : v)
    if (i)
      return false;
  return true;
}

bool cond(int i);

bool good_all_of() {
  int v[] = {1, 2, 3};
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: replace loop by 'std::ranges::all_of()'
  for (int i : v)
    if (!cond(i))
      return false;
  return true;
}
