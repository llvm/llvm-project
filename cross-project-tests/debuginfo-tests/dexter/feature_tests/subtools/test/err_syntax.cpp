// Purpose:
//      Check that parsing bad commands gives a useful error.
//          - Syntax error (misplaced ',')
//      Check directives are in check.txt to prevent dexter reading any embedded
//      commands.
//
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: not %dexter_base test --binary %t %dexter_regression_test_debugger_args \
// RUN:     -v -- %s | FileCheck %s --match-full-lines --strict-whitespace
//
// CHECK:parser error:{{.*}}err_syntax.cpp(18): invalid syntax
// CHECK:// {{Dex}}ExpectWatchValue(,'a', 3, 3, 3, 3, on_line=0)
// CHECK:                       ^

int main(){
    return 0;
}
// DexExpectWatchValue(,'a', 3, 3, 3, 3, on_line=0)
