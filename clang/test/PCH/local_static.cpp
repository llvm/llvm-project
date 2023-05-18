// REQUIRES: x86-registered-target
// Test this without PCH.
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9.0 -include %S/local_static.h -fsyntax-only %s -emit-llvm -o %t.no_pch.ll %s
// RUN: FileCheck --input-file %t.no_pch.ll %s

// Test with PCH.
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9.0 -x c++-header -emit-pch -o %t.pch %S/local_static.h
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9.0 -include-pch %t.pch -emit-llvm -o %t.pch.ll %s
// RUN: FileCheck --input-file %t.pch.ll %s

// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9.0 -x c++-header -emit-pch -fpch-instantiate-templates -o %t.pch %S/local_static.h
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9.0 -include-pch %t.pch -emit-llvm -o %t.pch.ll %s
// RUN: FileCheck --input-file %t.pch.ll %s

void test(Bar &b) {
  b.f<int>();
  static int s;
}

// Check if the mangling of static and local extern variables
// are correct and preserved by PCH.

// CHECK: @_ZZ4testR3BarE1s = internal global i32 0, align 4
// CHECK: @_ZZN3Bar1fIiEEvvE1y = linkonce_odr constant i32 0, align 4

