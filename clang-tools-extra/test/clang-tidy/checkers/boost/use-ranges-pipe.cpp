// RUN: %check_clang_tidy -std=c++14 %s boost-use-ranges %t -check-suffixes=,PIPE \
// RUN:   -config="{CheckOptions: { \
// RUN:     boost-use-ranges.UseReversePipe: true }}" -- -I %S/Inputs/use-ranges/
// RUN: %check_clang_tidy -std=c++14 %s boost-use-ranges %t -check-suffixes=,NOPIPE  -- -I %S/Inputs/use-ranges/

// CHECK-FIXES: #include <boost/algorithm/cxx11/is_sorted.hpp>
// CHECK-FIXES: #include <boost/range/adaptor/reversed.hpp>

#include "fake_std.h"

void stdLib() {
  std::vector<int> I;
  std::is_sorted_until(I.rbegin(), I.rend());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a boost version of this algorithm
  // CHECK-FIXES-NOPIPE: boost::algorithm::is_sorted_until(boost::adaptors::reverse(I));
  // CHECK-FIXES-PIPE: boost::algorithm::is_sorted_until(I | boost::adaptors::reversed);

}
