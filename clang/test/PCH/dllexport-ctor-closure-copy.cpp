// Test that MS ABI copy-constructor closures (built when throwing a class by
// value whose copy constructor has default arguments) survive a PCH
// round-trip. BuildCtorClosureDefaultArgs deliberately leaves the first
// default-arg slot null for a copy ctor closure (the closure itself supplies
// that argument).
//
// Test this without pch.
// RUN: %clang_cc1 -fcxx-exceptions -fms-extensions -triple i386-pc-win32 -std=c++20 -include %s -emit-llvm -o - %s | FileCheck %s
//
// Test with pch.
// RUN: %clang_cc1 -fcxx-exceptions -fms-extensions -triple i386-pc-win32 -std=c++20 -emit-pch -o %t %s
// RUN: %clang_cc1 -fcxx-exceptions -fms-extensions -triple i386-pc-win32 -std=c++20 -include-pch %t -emit-llvm -o - %s | FileCheck %s

#ifndef HEADER
#define HEADER

struct Default {
  Default(Default &, int = 42);
};

void h(Default &d) {
  throw d;
}

// CHECK-LABEL: define {{.*}} void @"?h@@YAXAAUDefault@@@Z"

#else

#endif
