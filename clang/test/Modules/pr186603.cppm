// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// Test that duplicate inline function definitions in different partitions
// of the same named module are diagnosed as ODR violations.
// See https://github.com/llvm/llvm-project/issues/186603
//
// Case 1: Module interface imports partition and redefines inline function
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/partition1.cppm -o %t/A-P1.pcm
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/module1.cppm \
// RUN:   -fmodule-file=A:P1=%t/A-P1.pcm -o %t/A.pcm -verify

//--- partition1.cppm
export module A:P1;
export inline void x() {}

//--- module1.cppm
export module A;
import :P1;
export inline void x() {} // expected-error {{redefinition of 'x'}}
// expected-note@partition1.cppm:* {{previous definition is here}}