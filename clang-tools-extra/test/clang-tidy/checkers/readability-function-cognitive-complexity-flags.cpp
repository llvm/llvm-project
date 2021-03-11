// RUN: %check_clang_tidy %s readability-function-cognitive-complexity %t -- \
// RUN:   -config='{CheckOptions: \
// RUN:             [{key: readability-function-cognitive-complexity.Threshold, \
// RUN:               value: 0}, \
// RUN:              {key: readability-function-cognitive-complexity.DescribeBasicIncrements, \
// RUN:               value: "false"} ]}'
// RUN: %check_clang_tidy -check-suffix=THRESHOLD5 %s readability-function-cognitive-complexity %t -- \
// RUN:   -config='{CheckOptions: \
// RUN:             [{key: readability-function-cognitive-complexity.Threshold, \
// RUN:               value: 5}, \
// RUN:              {key: readability-function-cognitive-complexity.DescribeBasicIncrements, \
// RUN:               value: "false"} ]}'

void func_of_complexity_4() {
  // CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'func_of_complexity_4' has cognitive complexity of 4 (threshold 0) [readability-function-cognitive-complexity]
  if (1) {
    if (1) {
    }
  }
  if (1) {
  }
}

#define MacroOfComplexity10 \
  if (1) {                  \
    if (1) {                \
      if (1) {              \
        if (1) {            \
        }                   \
      }                     \
    }                       \
  }

void function_with_macro() {
  // CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'function_with_macro' has cognitive complexity of 11 (threshold 0) [readability-function-cognitive-complexity]
  // CHECK-NOTES-THRESHOLD5: :[[@LINE-2]]:6: warning: function 'function_with_macro' has cognitive complexity of 11 (threshold 5) [readability-function-cognitive-complexity]

  MacroOfComplexity10;

  if (1) {
  }
}
