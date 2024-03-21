// Tests that we can use modules and ftime-trace together.
// Address https://github.com/llvm/llvm-project/issues/60544
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/a.pcm -ftime-trace=%t/a.json -o -
// RUN: ls %t | grep "a.json"

// Test again with reduced BMI.
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-reduced-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/a.pcm -ftime-trace=%t/a.json -o -
// RUN: ls %t | grep "a.json"

//--- a.cppm
export module a;
