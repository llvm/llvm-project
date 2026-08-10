// Test that we won't record local declarations by default in reduced BMI.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-reduced-module-interface -o %t/a.pcm
// RUN: llvm-bcanalyzer --dump --disable-histogram --show-binary-blobs %t/a.pcm > %t/a.dump
// RUN: cat %t/a.dump | FileCheck %t/a.cppm
//
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -emit-reduced-module-interface -o %t/b.pcm
// RUN: llvm-bcanalyzer --dump --disable-histogram --show-binary-blobs %t/b.pcm > %t/b.dump
// RUN: cat %t/b.dump | FileCheck %t/b.cppm

//--- a.cppm
export module a;
export int func() {
    int v = 43;
    return 43;
}

// Test that the variable declaration is not recorded completely.
// CHECK-NOT: <DECL_VAR

//--- b.cppm
export module b;
export inline int func() {
    int v = 43;
    return v;
}

// Check that we still records the declaration from inline functions.
// CHECK: <DECL_VAR
