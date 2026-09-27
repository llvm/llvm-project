// RUN: %check_clang_tidy -std=c++20 %s modernize-use-ranges %t -- -- -I %S/Inputs/
// RUN: %check_clang_tidy -std=c++23-or-later %s modernize-use-ranges %t -check-suffixes=,CPP23 -- -I %S/Inputs/

// CHECK-FIXES: #include <algorithm>
// CHECK-FIXES-CPP23: #include <numeric>
// CHECK-FIXES: #include <ranges>

#include "use-ranges/fake_std.h"
#include <memory>

void Positives() {
  std::vector<int> I, J;
  std::vector<std::unique_ptr<int>> K;

  // Expect to have no check messages
  std::find(K.begin(), K.end(), nullptr);

  std::find(K.begin(), K.end(), std::unique_ptr<int>());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::find(K, std::unique_ptr<int>());

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

  auto LogicalEnd = std::unique(I.begin(), I.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use a ranges version of this algorithm
  // CHECK-FIXES: auto LogicalEnd = std::ranges::unique(I).begin();

  bool AlreadyUnique = std::unique(I.begin(), I.end()) == I.end();
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: use a ranges version of this algorithm
  // CHECK-FIXES: bool AlreadyUnique = std::ranges::unique(I).begin() == I.end();

  auto LogicalEndWithPred =
      std::unique(I.begin(), I.end(), [](int A, int B) { return A == B; });
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use a ranges version of this algorithm
  // CHECK-FIXES: auto LogicalEndWithPred =
  // CHECK-FIXES-NEXT: std::ranges::unique(I, [](int A, int B) { return A == B; }).begin();

  std::unique(I.begin(), I.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::unique(I);

  I.erase(std::remove(I.begin(), I.end(), 0), I.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use a ranges version of this algorithm
  // CHECK-FIXES: I.erase(std::ranges::remove(I, 0).begin(), I.end());

  I.erase(std::remove_if(I.begin(), I.end(), [](int N) { return N == 0; }),
          I.end());
  // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: use a ranges version of this algorithm
  // CHECK-FIXES: I.erase(std::ranges::remove_if(I, [](int N) { return N == 0; }).begin(),

  auto PartitionPoint =
      std::partition(I.begin(), I.end(), [](int N) { return N == 0; });
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use a ranges version of this algorithm
  // CHECK-FIXES: auto PartitionPoint =
  // CHECK-FIXES-NEXT: std::ranges::partition(I, [](int N) { return N == 0; }).begin();

  auto StablePartitionPoint =
      std::stable_partition(I.begin(), I.end(), [](int N) { return N == 0; });
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use a ranges version of this algorithm
  // CHECK-FIXES: auto StablePartitionPoint =
  // CHECK-FIXES-NEXT: std::ranges::stable_partition(I, [](int N) { return N == 0; }).begin();

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

  auto RotatePoint = std::rotate(I.begin(), I.begin() + 2, I.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: use a ranges version of this algorithm
  // CHECK-FIXES: auto RotatePoint = std::ranges::rotate(I, I.begin() + 2).begin();

  using std::find;
  namespace my_std = std;

  // Potentially these could be updated to better qualify the replaced function name
  find(I.begin(), I.end(), 5);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::find(I, 5);

  using std::unique;
  auto LogicalEndFromUsing = unique(I.begin(), I.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: use a ranges version of this algorithm
  // CHECK-FIXES: auto LogicalEndFromUsing = std::ranges::unique(I).begin();

  my_std::find(I.begin(), I.end(), 6);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::find(I, 6);

  auto LogicalEndFromNamespaceAlias = my_std::unique(I.begin(), I.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: use a ranges version of this algorithm
  // CHECK-FIXES: auto LogicalEndFromNamespaceAlias = std::ranges::unique(I).begin();
}

void Reverse(){
  std::vector<int> I, J;
  std::vector<std::unique_ptr<int>> K;
  
  // Expect to have no check messages
  std::find(K.rbegin(), K.rend(), nullptr);

  std::find(K.rbegin(), K.rend(), std::unique_ptr<int>());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::find(std::views::reverse(K), std::unique_ptr<int>());

  std::find(I.rbegin(), I.rend(), 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::find(std::views::reverse(I), 0);

  auto ReverseLogicalEnd = std::unique(I.rbegin(), I.rend());
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use a ranges version of this algorithm
  // CHECK-FIXES: auto ReverseLogicalEnd = std::ranges::unique(std::views::reverse(I)).begin();

  std::equal(std::rbegin(I), std::rend(I), J.begin(), J.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::equal(std::views::reverse(I), J);

  std::equal(I.begin(), I.end(), std::crbegin(J), std::crend(J));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES: std::ranges::equal(I, std::views::reverse(J));
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
