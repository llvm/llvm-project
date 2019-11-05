// Purpose:
//     Check the `clang-opt-bisect` tool runs with typical input.
//
// REQUIRES: system-linux, lldb
//
// RUN: true
// RUN: %dexter_base clang-opt-bisect --debugger 'lldb' --builder 'clang' \
// RUN:     --cflags "-O0 -g" -- %s \
// RUN:     | FileCheck %s
// CHECK: running pass 0
// CHECK: wrote{{.*}}per_pass_score
// CHECK: wrote{{.*}}pass-summary
// CHECK: wrote{{.*}}overall-pass-summary

int main() {
    return 0;
}
