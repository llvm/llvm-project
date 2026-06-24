// RUN: %check_clang_tidy %s bugprone-container-bounds-check-overflow %t -- -config='{CheckOptions: { bugprone-container-bounds-check-overflow.IgnoredContainers: "::CustomClass"}}'

#include <cstddef>
#include <vector>
#include <string>

#define IS_WITHIN_BOUNDS(a, b, c) (a + b > c)
#define IS_WITHIN_BOUNDS2(a, b, c) ((a) + (b) > (c))

namespace {

class CustomClass {
public:
  size_t size() const {
    return 0;
  }
};

size_t size() {
  return 0;
}

// The check must work inside templates and across multiple instantiations
template <typename T>
void templated(size_t a, size_t b, const std::vector<T> &v) {
  if (a + b > v.size()) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: potential overflow in unsigned integer addition before comparison [bugprone-container-bounds-check-overflow]
  // CHECK-FIXES: if (a > v.size() || b > v.size() - a) {}
}

}

void positives(size_t a, size_t b, const std::vector<int> &v) {
  if (a + b > v.size()) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: potential overflow in unsigned integer addition before comparison [bugprone-container-bounds-check-overflow]
  // CHECK-FIXES: if (a > v.size() || b > v.size() - a) {}
  if (a + b >= v.size()) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: potential overflow in unsigned integer addition before comparison [bugprone-container-bounds-check-overflow]
  // CHECK-FIXES: if (a >= v.size() || b >= v.size() - a) {}
  if (a + b < v.size()) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: potential overflow in unsigned integer addition before comparison [bugprone-container-bounds-check-overflow]
  // CHECK-FIXES: if (a < v.size() && b < v.size() - a) {}
  if (a + b <= v.size()) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: potential overflow in unsigned integer addition before comparison [bugprone-container-bounds-check-overflow]
  // CHECK-FIXES: if (a <= v.size() && b <= v.size() - a) {}
  if (v.size() < a + b) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: potential overflow in unsigned integer addition before comparison [bugprone-container-bounds-check-overflow]
  // CHECK-FIXES: if (v.size() < a || v.size() - a < b) {}
  if (v.size() <= a + b) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: potential overflow in unsigned integer addition before comparison [bugprone-container-bounds-check-overflow]
  // CHECK-FIXES: if (v.size() <= a || v.size() - a <= b) {}
  if (v.size() > a + b) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: potential overflow in unsigned integer addition before comparison [bugprone-container-bounds-check-overflow]
  // CHECK-FIXES: if (v.size() > a && v.size() - a > b) {}
  if (v.size() >= a + b) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: potential overflow in unsigned integer addition before comparison [bugprone-container-bounds-check-overflow]
  // CHECK-FIXES: if (v.size() >= a && v.size() - a >= b) {}

  // Introduces parantheses to avoid changing the order of operations
  if (true && a + b > v.size()) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: potential overflow in unsigned integer addition before comparison [bugprone-container-bounds-check-overflow]
  // CHECK-FIXES: if (true && (a > v.size() || b > v.size() - a)) {}
  if (a + b > v.size() && true) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: potential overflow in unsigned integer addition before comparison [bugprone-container-bounds-check-overflow]
  // CHECK-FIXES: if ((a > v.size() || b > v.size() - a) && true) {}

  // Avoid introducing parantheses if the comparison is already wrapped in one
  while(a + b > v.size()) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: potential overflow in unsigned integer addition before comparison [bugprone-container-bounds-check-overflow]
  // CHECK-FIXES: while(a > v.size() || b > v.size() - a) {}
  auto result = a + b > v.size();
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: potential overflow in unsigned integer addition before comparison [bugprone-container-bounds-check-overflow]
  // CHECK-FIXES: auto result = (a > v.size() || b > v.size() - a);
  result = (a + b > v.size());
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: potential overflow in unsigned integer addition before comparison [bugprone-container-bounds-check-overflow]
  // CHECK-FIXES: result = (a > v.size() || b > v.size() - a);
  (void)result;

  // Confirm the fix works well with different named variables and container types
  size_t x = 1;
  size_t y = 2;
  std::string str = "Hello, world!";
  if (x + y > str.size()) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: potential overflow in unsigned integer addition before comparison [bugprone-container-bounds-check-overflow]
  // CHECK-FIXES: if (x > str.size() || y > str.size() - x) {}

  // The check should match local variables that are assigned the result of a size() call
  auto local_size = v.size();
  if (a + b < local_size) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: potential overflow in unsigned integer addition before comparison [bugprone-container-bounds-check-overflow]
  // CHECK-FIXES: if (a < local_size && b < local_size - a) {}
  if (local_size < a + b) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: potential overflow in unsigned integer addition before comparison [bugprone-container-bounds-check-overflow]
  // CHECK-FIXES: if (local_size < a || local_size - a < b) {}

  IS_WITHIN_BOUNDS(x, y, v.size());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: potential overflow in unsigned integer addition before comparison [bugprone-container-bounds-check-overflow]
  IS_WITHIN_BOUNDS2(x, y, v.size());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: potential overflow in unsigned integer addition before comparison [bugprone-container-bounds-check-overflow]

  // Chained additions: a compound operand of the addition produces a diagnostic only, with no fix
  size_t c = 10;
  if (a + b + c > v.size()) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: potential overflow in unsigned integer addition before comparison [bugprone-container-bounds-check-overflow]
  if (v.size() < a + b + c) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: potential overflow in unsigned integer addition before comparison [bugprone-container-bounds-check-overflow]
  if (a + (b + c) > v.size()) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: potential overflow in unsigned integer addition before comparison [bugprone-container-bounds-check-overflow]
  if (v.size() < a + b + c || v.empty()) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: potential overflow in unsigned integer addition before comparison [bugprone-container-bounds-check-overflow]
  if (!v.empty() && a + b + c > v.size()) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: potential overflow in unsigned integer addition before comparison [bugprone-container-bounds-check-overflow]

  // Instantiating templated function which contains the target pattern
  templated(a, b, std::vector<int>{});
  templated(a, b, std::vector<double>{});
}

void negatives(size_t a, size_t b, const std::vector<int> &v) {
  // Cannot overflow because of the comparison order
  if (a > v.size() || b > v.size() - a) {}
  if (b > v.size() || a > v.size() - b) {}

  // Cannot overflow because the operands of '+' are smaller than the result of size(); the addition result gets promoted to size_t before the comparison
  unsigned short x = 1;
  unsigned short y = 2;
  if (x + y > v.size()) {}

  // Intentionally ignored class
  CustomClass custom;
  if (a + b > custom.size()) {}
  if (a + b >= custom.size()) {}
  if (a + b < custom.size()) {}
  if (a + b <= custom.size()) {}
  if (custom.size() < a + b) {}
  if (custom.size() <= a + b) {}
  if (custom.size() > a + b) {}
  if (custom.size() >= a + b) {}

  // Call to a non-member size() method is not matched
  if (a + b > size()) {}

  // A plain variable (not a size() call) is not matched
  size_t n = 10;
  if (a + b > n) {}

  // Signed operands cannot overflow in a way this check cares about (signed overflow is UB), so they must not be matched
  int i = 1;
  int j = 2;
  if (i + j > v.size()) {}
  if (v.size() < i + j) {}
}

