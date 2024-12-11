

// RUN: %clang_cc1 -fbounds-safety -dump-tokens %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -dump-tokens %s 2>&1 | FileCheck %s

void Test() {
    (void) __builtin_unsafe_forge_terminated_by(int*, 1, 2);
    // CHECK: __builtin_unsafe_forge_terminated_by '__builtin_unsafe_forge_terminated_by'
    // CHECK: l_paren '('
    // CHECK: int 'int'
    // CHECK: star '*'
    // CHECK: comma ','
    // CHECK: numeric_constant '1'
    // CHECK: comma ','
    // CHECK: numeric_constant '2'
    // CHECK: r_paren ')'
}
