// Make SEH works in PCH
//
// Test this without pch.
// RUN: %clang_cc1 -fms-extensions -triple x86_64-windows-msvc -std=c++11 -include %s -emit-llvm -o - %s | FileCheck %s

// Test with pch.
// RUN: %clang_cc1 -fms-extensions -triple x86_64-windows-msvc -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fms-extensions -triple x86_64-windows-msvc -std=c++11 -include-pch %t -emit-llvm -o - %s | FileCheck %s

#ifndef HEADER
#define HEADER

int shouldCatch();
inline int f() {
  __try {
  } __except (shouldCatch()) {
  }
  return 0;
}
int x = f();

// CHECK: define linkonce_odr dso_local i32 @"?f@@YAHXZ"()
// CHECK: define internal i32 @"?filt$0@0@f@@"({{.*}})

#else

// empty

#endif
