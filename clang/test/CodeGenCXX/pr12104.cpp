// RUN: %clang_cc1 -include %S/pr12104.h %s -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -x c++ -triple %itanium_abi_triple -emit-pch -o %t %S/pr12104.h
// RUN: %clang_cc1 -include-pch %t %s -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %s

template struct Patch<1>;

// CHECK: _ZN5PatchILi1EE11no_neighborE
