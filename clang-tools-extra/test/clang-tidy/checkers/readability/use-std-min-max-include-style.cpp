// RUN: %check_clang_tidy -std=c++11-or-later %s readability-use-std-min-max %t \
// RUN: -config='{CheckOptions: {readability-use-std-min-max.IncludeStyle: "google"}}' \
// RUN: -- -fno-delayed-template-parsing

// CHECK-FIXES: #include <algorithm>

void foo() {
  int a = 0, b = 1;
  if (a < b)
    a = b;
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use `std::max` instead of `<` [readability-use-std-min-max]
  // CHECK-FIXES: a = std::max(a, b);
}
