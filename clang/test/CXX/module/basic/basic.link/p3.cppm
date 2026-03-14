// RUN: %clang_cc1 -std=c++20 -triple x86_64-linux %s -emit-module-interface -o %t
// RUN: %clang_cc1 -std=c++20 -triple x86_64-linux -x pcm %t -emit-llvm -o - | FileCheck %s

export module M;

// CHECK: @_ZW1M1a ={{.*}} constant i32 1
const int a = 1;
// CHECK: @_ZW1M1b ={{.*}} constant i32 2
export const int b = 2;

export int f() { return a + b; }
