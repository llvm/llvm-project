// Test that the static function only used in non-inline functions won't get emitted
// into the BMI.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
//
// RUN: %clang_cc1 -std=c++20 %s -emit-reduced-module-interface -o %t/S.pcm
// RUN: llvm-bcanalyzer --dump --disable-histogram --show-binary-blobs %t/S.pcm > %t/S.dump
// RUN: cat %t/S.dump | FileCheck %s

export module S;
static int static_func() {
    return 43;
}

export int func() {
    return static_func();
}

// CHECK: <DECL_FUNCTION
// Checks that we won't see a second function
// CHECK-NOT: <DECL_FUNCTION
