// Purpose:
//    Check that \DexDeclareFile causes a DexExpectWatchValue's to generate a
//    missing value penalty when the declared path is incorrect.
//
// UNSUPPORTED: system-darwin
//
//
// RUN: %dexter_regression_test_cxx_build %s -o %t
// RUN: not %dexter_regression_test_run --binary %t -- %s | FileCheck %s
// CHECK: dex_declare_file.cpp

int main() {
  int result = 0;
  return result; //DexLabel('return')
}

// DexDeclareFile('this_file_does_not_exist.cpp')
// DexExpectWatchValue('result', 0, on_line='return')
