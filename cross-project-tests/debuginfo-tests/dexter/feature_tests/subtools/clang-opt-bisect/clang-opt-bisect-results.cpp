// Purpose:
//     Check the `clang-opt-bisect` tool runs with --results-directory.
//
// RUN: true
// RUN: %dexter_base clang-opt-bisect \
// RUN:     --debugger %dexter_regression_test_debugger \
// RUN:     --builder %dexter_regression_test_builder \
// RUN:     --cflags "%dexter_regression_test_cflags" \
// RUN:     --ldflags "%dexter_regression_test_ldflags" \
// RUN:     --results-directory=%t \
// RUN:     -- %s \
// RUN: | FileCheck %s
//// Clean up those results files.
// RUN: rm %t/clang-opt-bisect-results.cpp-pass-summary.csv
// RUN: rm %t/clang-opt-bisect-results.cpp-per_pass_score.csv
// RUN: rm %t/overall-pass-summary.csv
// RUN: rm %t/*.dextIR
// RUN: rm %t/*.txt
// RUN: rmdir %t
// CHECK: running pass 0
// CHECK: wrote{{.*}}per_pass_score
// CHECK: wrote{{.*}}pass-summary
// CHECK: wrote{{.*}}overall-pass-summary

int main() {
    return 0;
}
