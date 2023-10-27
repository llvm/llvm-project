// Purpose:
//      Check that using an undeclared address gives a useful error message.
//
// RUN: %dexter_regression_test_build %s -o %t
// RUN: not %dexter_regression_test_run --binary %t -v -- %s | FileCheck %s --match-full-lines


int main() {
    int *result = new int(0);
    delete result; // DexLabel('test_line')
}


// CHECK: parser error:{{.*}}err_undeclared_addr.cpp([[# @LINE + 3]]): Undeclared address: 'result'
// CHECK-NEXT: {{Dex}}ExpectWatchValue('result', address('result'), on_line=ref('test_line'))

// DexExpectWatchValue('result', address('result'), on_line=ref('test_line'))
