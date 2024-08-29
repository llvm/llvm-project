// Test that the static function only used in non-inline functions won't get emitted
// into the BMI.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/S.cppm -emit-reduced-module-interface -o %t/S.pcm
// RUN: llvm-bcanalyzer --dump --disable-histogram --show-binary-blobs %t/S.pcm > %t/S.dump
// RUN: cat %t/S.dump | FileCheck %t/S.check

//--- S.cppm
export module S;
static int static_func() {
    return 43;
}

export int func() {
    return static_func();
}

//--- S.check
// CHECK: <DECL_FUNCTION
// Checks that we won't see a second function
// CHECK-NOT: <DECL_FUNCTION
