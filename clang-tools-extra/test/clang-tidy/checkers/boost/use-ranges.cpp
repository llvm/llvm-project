// RUN: %check_clang_tidy -std=c++14 %s boost-use-ranges %t  -- -- -I %S/Inputs/use-ranges/
// RUN: %check_clang_tidy -std=c++17 %s boost-use-ranges %t -check-suffixes=,CPP17 -- -I %S/Inputs/use-ranges/

// CHECK-FIXES: #include <boost/range/algorithm/find.hpp>
// CHECK-FIXES: #include <boost/range/algorithm/reverse.hpp>
// CHECK-FIXES: #include <boost/range/algorithm/set_algorithm.hpp>
// CHECK-FIXES: #include <boost/range/algorithm/equal.hpp>
// CHECK-FIXES: #include <boost/range/algorithm/permutation.hpp>
// CHECK-FIXES: #include <boost/range/algorithm/heap_algorithm.hpp>
// CHECK-FIXES: #include <boost/algorithm/cxx11/copy_if.hpp>
// CHECK-FIXES: #include <boost/algorithm/cxx11/is_sorted.hpp>
// CHECK-FIXES-CPP17: #include <boost/algorithm/cxx17/reduce.hpp>
// CHECK-FIXES: #include <boost/range/adaptor/reversed.hpp>
// CHECK-FIXES: #include <boost/range/numeric.hpp>

#include "fake_boost.h"
#include "fake_std.h"

bool returnTrue(int val) {
  return true;
}

void stdLib() {
  std::vector<int> I, J;
  std::find(I.begin(), I.end(), 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a boost version of this algorithm
  // CHECK-FIXES: boost::range::find(I, 0);

  std::reverse(I.cbegin(), I.cend());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a boost version of this algorithm
  // CHECK-FIXES: boost::range::reverse(I);

  std::includes(I.begin(), I.end(), std::begin(J), std::end(J));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a boost version of this algorithm
  // CHECK-FIXES: boost::range::includes(I, J);

  std::equal(std::cbegin(I), std::cend(I), J.begin(), J.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a boost version of this algorithm
  // CHECK-FIXES: boost::range::equal(I, J);

  std::next_permutation(I.begin(), I.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a boost version of this algorithm
  // CHECK-FIXES: boost::range::next_permutation(I);

  std::push_heap(I.begin(), I.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a boost version of this algorithm
  // CHECK-FIXES: boost::range::push_heap(I);

  std::copy_if(I.begin(), I.end(), J.begin(), &returnTrue);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a boost version of this algorithm
  // CHECK-FIXES: boost::algorithm::copy_if(I, J.begin(), &returnTrue);

  std::is_sorted_until(I.begin(), I.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a boost version of this algorithm
  // CHECK-FIXES: boost::algorithm::is_sorted_until(I);

  std::reduce(I.begin(), I.end());
  // CHECK-MESSAGES-CPP17: :[[@LINE-1]]:3: warning: use a boost version of this algorithm
  // CHECK-FIXES-CPP17: boost::algorithm::reduce(I);

  std::reduce(I.begin(), I.end(), 2);
  // CHECK-MESSAGES-CPP17: :[[@LINE-1]]:3: warning: use a boost version of this algorithm
  // CHECK-FIXES-CPP17: boost::algorithm::reduce(I, 2);

  std::reduce(I.begin(), I.end(), 0, [](int a, int b){ return a + b; });
  // CHECK-MESSAGES-CPP17: :[[@LINE-1]]:3: warning: use a boost version of this algorithm
  // CHECK-FIXES-CPP17: boost::algorithm::reduce(I, 0, [](int a, int b){ return a + b; });

  std::equal(boost::rbegin(I), boost::rend(I), J.begin(), J.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a boost version of this algorithm
  // CHECK-FIXES: boost::range::equal(boost::adaptors::reverse(I), J);

  std::accumulate(I.begin(), I.end(), 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a boost version of this algorithm
  // CHECK-FIXES: boost::accumulate(I, 0);
}

void boostLib() {
  std::vector<int> I;
  boost::algorithm::reduce(I.begin(), I.end(), 0, [](int a, int b){ return a + b; });
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranged version of this algorithm
  // CHECK-FIXES: boost::algorithm::reduce(I, 0, [](int a, int b){ return a + b; });

  boost::algorithm::reduce(boost::begin(I), boost::end(I), 1, [](int a, int b){ return a + b; });
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranged version of this algorithm
  // CHECK-FIXES: boost::algorithm::reduce(I, 1, [](int a, int b){ return a + b; });

  boost::algorithm::reduce(boost::const_begin(I), boost::const_end(I), 2, [](int a, int b){ return a + b; });
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranged version of this algorithm
  // CHECK-FIXES: boost::algorithm::reduce(I, 2, [](int a, int b){ return a + b; });
}
