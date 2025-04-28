// Purpose:
//      Check that parsing bad commands gives a useful error.
//          - Unbalanced parenthesis
//      Check directives are in check.txt to prevent dexter reading any embedded
//      commands.
//
// Note: Despite using 'lldb' as the debugger, lldb is not actually required
//       as the test should finish before lldb would be invoked.
//
// RUN: %dexter_regression_test_build %s -o %t
// RUN: not %dexter_base test --binary %t --debugger 'lldb' \
// RUN:     -v -- %s | FileCheck %s --match-full-lines --strict-whitespace
//
// CHECK:parser error:{{.*}}err_paren.cpp(22): Unbalanced parenthesis starting here
// CHECK:// {{Dex}}ExpectWatchValue(
// CHECK:                      ^

int main(){
    return 0;
}

// DexExpectWatchValue(
