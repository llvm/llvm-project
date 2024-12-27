// RUN: %check_clang_tidy -std=c++20 %s modernize-use-ranges %t -check-suffixes=,PIPE \
// RUN:   -config="{CheckOptions: { \
// RUN:     modernize-use-ranges.UseReversePipe: true }}" -- -I %S/Inputs/use-ranges/
// RUN: %check_clang_tidy -std=c++20 %s modernize-use-ranges %t -check-suffixes=,NOPIPE  -- -I %S/Inputs/use-ranges/

// CHECK-FIXES: #include <algorithm>
// CHECK-FIXES: #include <ranges>

#include "fake_std.h"

void stdLib() {
  std::vector<int> I;
  std::find(I.rbegin(), I.rend(), 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a ranges version of this algorithm
  // CHECK-FIXES-NOPIPE: std::ranges::find(std::ranges::reverse_view(I), 0);
  // CHECK-FIXES-PIPE: std::ranges::find(I | std::views::reverse, 0);

}
