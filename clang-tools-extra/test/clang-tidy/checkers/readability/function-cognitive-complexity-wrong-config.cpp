// RUN: %check_clang_tidy %s readability-function-cognitive-complexity %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     readability-function-cognitive-complexity.Threshold: 10, \
// RUN:     readability-function-cognitive-complexity.IgnoreAboveThreshold: 5, \
// RUN:     readability-function-cognitive-complexity.DescribeBasicIncrements: false \
// RUN:   }}'

// CHECK-MESSAGES: warning: 'IgnoreAboveThreshold' option value '5' is less than 'Threshold' option value '10'; the option will be ignored [clang-tidy-config]

// The IgnoreAboveThreshold option is ignored when it's less than Threshold,
// so this function with complexity 11 should still be flagged.
void func_of_complexity_11() {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'func_of_complexity_11' has cognitive complexity of 11 (threshold 10) [readability-function-cognitive-complexity]
  if (1) {
    if (1) {
      if (1) {
        if (1) {
        }
      }
    }
  }
}
