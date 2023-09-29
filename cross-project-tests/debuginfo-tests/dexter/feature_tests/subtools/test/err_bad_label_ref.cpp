// Purpose:
//      Check that referencing an undefined label gives a useful error message.
//
// RUN: %dexter_regression_test_build %s -o %t
// RUN: not %dexter_regression_test_run --binary %t -v -- %s | FileCheck %s --match-full-lines
//
// CHECK: parser error:{{.*}}err_bad_label_ref.cpp(15): Unresolved label: 'label_does_not_exist'
// CHECK-NEXT: {{Dex}}ExpectWatchValue('result', '0', on_line=ref('label_does_not_exist'))

int main() {
    int result = 0;
    return result;
}

// DexExpectWatchValue('result', '0', on_line=ref('label_does_not_exist'))
