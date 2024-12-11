

// RUN: %clang_cc1 -fbounds-safety -dump-tokens %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -dump-tokens %s 2>&1 | FileCheck %s

void Test() {
    (void) __builtin_unsafe_forge_single(0);
    // CHECK: __builtin_unsafe_forge_single '__builtin_unsafe_forge_single'
}
