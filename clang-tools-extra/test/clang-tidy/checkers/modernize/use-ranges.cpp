// RUN: %check_clang_tidy -std=c++20 %s modernize-use-ranges %t -- -- -I %S/Inputs/use-ranges/
// RUN: %check_clang_tidy -std=c++23 %s modernize-use-ranges %t -check-suffixes=,CPP23 -- -I %S/Inputs/use-ranges/

// CHECK-FIXES: #include <algorithm>
// CHECK-FIXES-CPP23: #include <numeric>
// CHECK-FIXES: #include <ranges>

#include "fake_std.h"

void Positives() {
  std::vector<int> I, J;
  std::find(I.begin(), I.end(), 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::find(I, 0);

  std::find(I.cbegin(), I.cend(), 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::find(I, 1);

  std::find(std::begin(I), std::end(I), 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::find(I, 2);

  std::find(std::cbegin(I), std::cend(I), 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::find(I, 3);

  std::find(std::cbegin(I), I.cend(), 4);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::find(I, 4);

  std::reverse(I.begin(), I.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::reverse(I);

  std::includes(I.begin(), I.end(), I.begin(), I.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::includes(I, I);

  std::includes(I.begin(), I.end(), J.begin(), J.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::includes(I, J);

  std::is_permutation(I.begin(), I.end(), J.begin(), J.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::is_permutation(I, J);

  std::equal(I.begin(), I.end(), J.begin(), J.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::equal(I, J);

  std::equal(I.begin(), I.end(), J.begin(), J.end(), [](int a, int b){ return a == b; });
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::equal(I, J, [](int a, int b){ return a == b; });

  std::iota(I.begin(), I.end(), 0);
  // CHECK-MESSAGES-CPP23: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES-CPP23: std::ranges::iota(I, 0);

  std::rotate(I.begin(), I.begin() + 2, I.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::rotate(I, I.begin() + 2);

  using std::find;
  namespace my_std = std;

  // Potentially these could be updated to better qualify the replaced function name
  find(I.begin(), I.end(), 5);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::find(I, 5);

  my_std::find(I.begin(), I.end(), 6);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::find(I, 6);
}

void Reverse(){
  std::vector<int> I, J;
  std::find(I.rbegin(), I.rend(), 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::find(std::ranges::reverse_view(I), 0);

  std::equal(std::rbegin(I), std::rend(I), J.begin(), J.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::equal(std::ranges::reverse_view(I), J);

  std::equal(I.begin(), I.end(), std::crbegin(J), std::crend(J));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::equal(I, std::ranges::reverse_view(J));
}

void Negatives() {
  std::vector<int> I, J;
  std::find(I.begin(), J.end(), 0);
  std::find(I.begin(), I.begin(), 0);
  std::find(I.end(), I.begin(), 0);


  // Need both ranges for this one
  std::is_permutation(I.begin(), I.end(), J.begin());

  // We only have one valid match here and the ranges::equal function needs 2 complete ranges
  std::equal(I.begin(), I.end(), J.begin());
  std::equal(I.begin(), I.end(), J.end(), J.end());
  std::equal(std::rbegin(I), std::rend(I), std::rend(J), std::rbegin(J));
  std::equal(I.begin(), J.end(), I.begin(), I.end());

  // std::rotate expects the full range in the 1st and 3rd argument.
  // Anyone writing this code has probably written a bug, but this isn't the
  // purpose of this check.
  std::rotate(I.begin(), I.end(), I.begin() + 2);
  // Pathological, but probably shouldn't diagnose this
  std::rotate(I.begin(), I.end(), I.end() + 0);
}
