// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-reduced-module-interface -o %t/a.reduced.pcm
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -fexperimental-modules-reduced-bmi -fmodule-output=%t/a.pcm \
// RUN:     -emit-llvm -o %t/a.ll
//
// Test that the generated BMI from `-fexperimental-modules-reduced-bmi -fmodule-output=` is same with
// `-emit-reduced-module-interface`.
// RUN: diff %t/a.reduced.pcm %t/a.pcm
//
// Test that we can consume the produced BMI correctly.
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -fmodule-file=a=%t/a.pcm -fsyntax-only -verify
//
// RUN: rm -f %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -fexperimental-modules-reduced-bmi -fmodule-output=%t/a.pcm \
// RUN:     -emit-module-interface -o %t/a.full.pcm
// RUN: diff %t/a.reduced.pcm %t/a.pcm
// RUN: not diff %t/a.pcm %t/a.full.pcm
//
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -fmodule-file=a=%t/a.pcm -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -fmodule-file=a=%t/a.full.pcm -fsyntax-only -verify

//--- a.cppm
export module a;
export int a() {
    return 43;
}

//--- b.cppm
// Test that we can consume the produced BMI correctly as a smocking test.
// expected-no-diagnostics
export module b;
import a;
export int b() { return a(); }
