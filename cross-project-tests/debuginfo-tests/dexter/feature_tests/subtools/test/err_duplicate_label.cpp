// Purpose:
//      Check that defining duplicate labels gives a useful error message.
//
// RUN: not %dexter_regression_test -v -- %s | FileCheck %s --match-full-lines
//
// CHECK: parser error:{{.*}}err_duplicate_label.cpp(11): Found duplicate line label: 'oops'
// CHECK-NEXT: {{Dex}}Label('oops')

int main() {
    int result = 0; // DexLabel('oops')
    return result;  // DexLabel('oops')
}
