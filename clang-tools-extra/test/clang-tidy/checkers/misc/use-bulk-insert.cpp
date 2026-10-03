// RUN: %check_clang_tidy %s misc-use-bulk-insert %t

#include <set>
#include <vector>

void test_set(const std::set<int> &In) {
  std::set<int> Out;

  for (int I : In) {
    Out.insert(I);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use bulk insertion instead of inserting elements one at a time [misc-use-bulk-insert]
    // CHECK-FIXES: Out.insert(In.begin(), In.end());
  }
}

void test_vector(const std::vector<int> &In) {
  std::vector<int> Out;

  for (int I : In) {
    Out.insert(I);
  }
}
