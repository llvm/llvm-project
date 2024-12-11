

// RUN: %clang_cc1 -fbounds-safety -dump-tokens %s 2>&1 | FileCheck %s

void Test() {
    (void) __builtin_unsafe_forge_bidi_indexable(0, sizeof(int));
    // CHECK: __builtin_unsafe_forge_bidi_indexable '__builtin_unsafe_forge_bidi_indexable'
}
