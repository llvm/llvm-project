// Make sure that the declarations inside the language linkage can
// be generated correctly.
//
// RUN: rm -fr %t
// RUN: mkdir %t
//
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %s -emit-module-interface -o %t/M.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/M.pcm -S -emit-llvm -disable-llvm-passes -o - | FileCheck %s
export module M;

extern "C++" {
void foo() {} 
}

extern "C" void bar() {}

// CHECK: define {{.*}}@bar(
// CHECK: define {{.*}}@_Z3foov(
