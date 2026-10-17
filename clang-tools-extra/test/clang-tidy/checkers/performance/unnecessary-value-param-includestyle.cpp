// RUN: %check_clang_tidy %s performance-unnecessary-value-param %t -- \
// RUN:   -config="{CheckOptions: {performance-unnecessary-value-param.IncludeStyle: 'google' \
// RUN: }}" -- -fno-delayed-template-parsing

// CHECK-FIXES: #include <utility>

#include <utility>

struct ExpensiveMovableType {
  ExpensiveMovableType();
  ExpensiveMovableType(ExpensiveMovableType &&);
  ExpensiveMovableType(const ExpensiveMovableType &) = default;
  ExpensiveMovableType &operator=(const ExpensiveMovableType &) = default;
  ExpensiveMovableType &operator=(ExpensiveMovableType &&);
  ~ExpensiveMovableType();
};

void PositiveMoveOnCopyConstruction(ExpensiveMovableType E) {
  auto F = E;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: parameter 'E' of type 'ExpensiveMovableType' is passed by value and only copied once; consider moving it to avoid unnecessary copies [performance-unnecessary-value-param]
  // CHECK-FIXES: auto F = std::move(E);
}
