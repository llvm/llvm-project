// Purpose:
//      Check that Dexter command syntax errors associate with the line and file
//      they appeared in rather than the current declared file.
//
// RUN: %dexter_regression_test_build %s -o %t
// RUN: not %dexter_base test --binary %t --debugger 'lldb' -v -- %s \
// RUN:     | FileCheck %s --implicit-check-not=FAIL-FILENAME-MATCH

// CHECK: err_syntax_dexdeclarefile.cpp(14): Undeclared address: 'not_been_declared'

int main() { return 0; }

// DexDeclareFile('FAIL-FILENAME-MATCH')
// DexExpectWatchValue('example', address('not_been_declared'))
