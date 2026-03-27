// Purpose:
//    Check that \DexDeclareFile changes the path of all succeeding commands
//    to the file path it declares. Also check that dexter correctly accepts
//    files with .dex extensions.
//
// UNSUPPORTED: system-darwin
//
//
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: %dexter_regression_test_run --binary %t -- %s | FileCheck %s
// CHECK: dex_and_source

int main() {
  int result = 0;
  return result;
}
