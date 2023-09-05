// Purpose:
//     Check the `clang-opt-bisect` tool runs with typical input.
//
// RUN: true
// RUN: %dexter_base clang-opt-bisect \
// RUN:     --debugger %dexter_regression_test_debugger \
// RUN:     --builder %dexter_regression_test_builder \
// RUN:     --cflags "%dexter_regression_test_cflags" \
// RUN:     --ldflags "%dexter_regression_test_ldflags" \
// RUN:     -- %s \
// RUN: | FileCheck %s
// CHECK: running pass 0

int main() {
    return 0;
}
