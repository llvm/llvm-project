// Purpose:
//      Check that parsing bad commands gives a useful error.
//          - Syntax error (misplaced ',') over multiple lines
//      Check directives are in check.txt to prevent dexter reading any embedded
//      commands.
//
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: not %dexter_base test --binary %t %dexter_regression_test_debugger_args \
// RUN:     -v -- %s | FileCheck --dump-input-context=999999999 %s --match-full-lines --strict-whitespace
//
// CHECK:parser error:{{.*}}err_syntax_mline.cpp(21): invalid syntax
// CHECK:    ,'a', 3, 3, 3, 3, on_line=0)
// CHECK:    ^

int main(){
    return 0;
}

/*
DexExpectWatchValue(
    ,'a', 3, 3, 3, 3, on_line=0)
*/
