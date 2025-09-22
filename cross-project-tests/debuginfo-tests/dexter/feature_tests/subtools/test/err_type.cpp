// Purpose:
//      Check that parsing bad commands gives a useful error.
//          - Type error (missing args)
//      Check directives are in check.txt to prevent dexter reading any embedded
//      commands.
//
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: not %dexter_base test --binary %t %dexter_regression_test_debugger_args \
// RUN:     -v -- %s | FileCheck %s --match-full-lines --strict-whitespace
//
// CHECK:parser error:{{.*}}err_type.cpp(18): expected at least two args
// CHECK:// {{Dex}}ExpectWatchValue()
// CHECK:                      ^

int main(){
    return 0;
}
// DexExpectWatchValue()
