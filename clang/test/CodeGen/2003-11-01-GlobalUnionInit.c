// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

union bdflush_param {
    struct { int x; } b_un;
    int y[1];
} bdf_prm = {{30}};

