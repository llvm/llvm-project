// RUN: %check_clang_tidy -std=c++17-or-later %s modernize-use-structured-binding %t \
// RUN: -config="{CheckOptions: {modernize-use-structured-binding.PairTypes: 'custom::pair; otherPair'}}"

namespace custom {
  struct pair {
    int first;
    int second;
  };
}

struct otherPair {
  int first;
  int second;
};

void OptionTest() {
  {
    auto P = custom::pair();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: auto [x, y] = custom::pair();
    int x = P.first;
    int y = P.second;
  }

  {
    auto P = otherPair();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: auto [x, y] = otherPair();
    int x = P.first;
    int y = P.second;
  }
}
