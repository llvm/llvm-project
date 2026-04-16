// RUN: %check_clang_tidy %s hicpp-function-size %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     hicpp-function-size.StatementThreshold: 0, \
// RUN:     hicpp-function-size.BranchThreshold: 100, \
// RUN:     hicpp-function-size.ParameterThreshold: 100, \
// RUN:     hicpp-function-size.NestingThreshold: 100, \
// RUN:     hicpp-function-size.VariableThreshold: 100 \
// RUN:   }}'

void f() { ; }
// CHECK-MESSAGES: warning: 'hicpp-function-size' check is deprecated and will be removed in a future release; consider using 'readability-function-size' instead [clang-tidy-config]
// CHECK-MESSAGES: :[[@LINE-2]]:6: warning: function 'f' exceeds recommended size/complexity thresholds [hicpp-function-size]
// CHECK-MESSAGES: :[[@LINE-3]]:6: note: 1 statements (threshold 0)
