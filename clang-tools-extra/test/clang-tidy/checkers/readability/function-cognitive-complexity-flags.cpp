// RUN: %check_clang_tidy %s readability-function-cognitive-complexity %t -- \
// RUN:   -config='{CheckOptions: \
// RUN:             {readability-function-cognitive-complexity.Threshold: 0, \
// RUN:              readability-function-cognitive-complexity.DescribeBasicIncrements: "false"}}'
// RUN: %check_clang_tidy -check-suffix=THRESHOLD5 %s readability-function-cognitive-complexity %t -- \
// RUN:   -config='{CheckOptions: \
// RUN:             {readability-function-cognitive-complexity.Threshold: 5, \
// RUN:              readability-function-cognitive-complexity.DescribeBasicIncrements: "false"}}'
// RUN: %check_clang_tidy -check-suffix=IGNORE-MACROS %s readability-function-cognitive-complexity %t -- \
// RUN:   -config='{CheckOptions: \
// RUN:             {readability-function-cognitive-complexity.Threshold: 0, \
// RUN:              readability-function-cognitive-complexity.IgnoreMacros: "true", \
// RUN:              readability-function-cognitive-complexity.DescribeBasicIncrements: "false"}}'

void func_of_complexity_4() {
  // CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'func_of_complexity_4' has cognitive complexity of 4 (threshold 0) [readability-function-cognitive-complexity]
  // CHECK-NOTES-IGNORE-MACROS: :[[@LINE-2]]:6: warning: function 'func_of_complexity_4' has cognitive complexity of 4 (threshold 0) [readability-function-cognitive-complexity]
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
  // CHECK-NOTES-IGNORE-MACROS: :[[@LINE-3]]:6: warning: function 'function_with_macro' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]

  MacroOfComplexity10;

  if (1) {
  }
}

#define noop \
  {}

#define SomeMacro(x) \
  if (1) {           \
    x;               \
  }

void func_macro_1() {
  // CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'func_macro_1' has cognitive complexity of 2 (threshold 0) [readability-function-cognitive-complexity]
  // CHECK-NOTES-IGNORE-MACROS: :[[@LINE-2]]:6: warning: function 'func_macro_1' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]

  if (1) {
  }
  SomeMacro(noop);
}

void func_macro_2() {
  // CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'func_macro_2' has cognitive complexity of 4 (threshold 0) [readability-function-cognitive-complexity]
  // CHECK-NOTES-IGNORE-MACROS: :[[@LINE-2]]:6: warning: function 'func_macro_2' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]

  if (1) {
  }
  // Note that if the IgnoreMacro option is set to 'true', currently also macro
  // arguments are ignored. Optimally, macros should be treated like function
  // calls, i.e. the arguments account to the complexity so that the overall
  // complexity of this function is 2 (1 for the if statement above + 1 for
  // the if statement in the argument).
  SomeMacro(if (1) { noop; });
}
