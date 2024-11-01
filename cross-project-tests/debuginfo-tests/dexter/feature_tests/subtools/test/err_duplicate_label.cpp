// Purpose:
//      Check that defining duplicate labels gives a useful error message.
//
// RUN: %dexter_regression_test_build %s -o %t
// RUN: not %dexter_regression_test_run --binary %t -v -- %s | FileCheck %s --match-full-lines
//
// CHECK: parser error:{{.*}}err_duplicate_label.cpp(12): Found duplicate line label: 'oops'
// CHECK-NEXT: {{Dex}}Label('oops')

int main() {
    int result = 0; // DexLabel('oops')
    return result;  // DexLabel('oops')
}
