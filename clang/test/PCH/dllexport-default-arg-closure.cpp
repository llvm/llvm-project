// Make sure we emit the MS ABI default ctor closure with PCH.
//
// Test this without pch.
// RUN: %clang_cc1 -no-opaque-pointers -fms-extensions -triple x86_64-windows-msvc -std=c++11 -include %s -emit-llvm -o - %s | FileCheck %s

// Test with pch.
// RUN: %clang_cc1 -no-opaque-pointers -fms-extensions -triple x86_64-windows-msvc -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fms-extensions -triple x86_64-windows-msvc -std=c++11 -include-pch %t -emit-llvm -o - %s | FileCheck %s

#ifndef HEADER
#define HEADER

struct __declspec(dllexport) Foo {
  enum E { E0 } e;
  Foo(E e = E0) : e(e) {}
};

// Demangles as:
// void Foo::`default constructor closure'(void)
// CHECK: define weak_odr dso_local dllexport void @"??_FFoo@@QEAAXXZ"(%struct.Foo*{{.*}})
// CHECK:   call noundef %struct.Foo* @"??0Foo@@QEAA@W4E@0@@Z"(%struct.Foo* {{.*}}, i32 noundef 0)

#else


#endif
