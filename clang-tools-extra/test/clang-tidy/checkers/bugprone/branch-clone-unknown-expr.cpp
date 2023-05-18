// RUN: %check_clang_tidy -expect-clang-tidy-error %s bugprone-branch-clone %t

int test_unknown_expression() {
  if (unknown_expression_1) {        // CHECK-MESSAGES: :[[@LINE]]:7: error: use of undeclared identifier 'unknown_expression_1' [clang-diagnostic-error]
    function1(unknown_expression_2); // CHECK-MESSAGES: :[[@LINE]]:15: error: use of undeclared identifier 'unknown_expression_2' [clang-diagnostic-error]
  } else {
    function2(unknown_expression_3); // CHECK-MESSAGES: :[[@LINE]]:15: error: use of undeclared identifier 'unknown_expression_3' [clang-diagnostic-error]
  }
}
