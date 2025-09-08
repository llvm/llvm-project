// Purpose:
//      Check that declaring duplicate addresses gives a useful error message.
//
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: not %dexter_regression_test_run --binary %t -v -- %s | FileCheck --dump-input-context=999999999 %s --match-full-lines


int main() {
    int *result = new int(0);
    delete result; // DexLabel('test_line')
}

// CHECK: parser error:{{.*}}err_duplicate_address.cpp([[# @LINE + 4]]): Found duplicate address: 'oops'
// CHECK-NEXT: {{Dex}}DeclareAddress('oops', 'result', on_line=ref('test_line'))

// DexDeclareAddress('oops', 'result', on_line=ref('test_line'))
// DexDeclareAddress('oops', 'result', on_line=ref('test_line'))
