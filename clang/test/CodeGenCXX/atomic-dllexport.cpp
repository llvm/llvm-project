// RUN: %clang_cc1 -no-opaque-pointers -triple i686-windows-msvc -emit-llvm -std=c++11 -fms-extensions -O0 -o - %s | FileCheck --check-prefix=M32 %s
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-windows-msvc -emit-llvm -std=c++11 -fms-extensions -O0 -o - %s | FileCheck --check-prefix=M64 %s

struct __declspec(dllexport) SomeStruct {
  // Copy assignment operator should be produced, and exported:
  // M32: define weak_odr dso_local dllexport x86_thiscallcc noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %struct.SomeStruct* @"??4SomeStruct@@QAEAAU0@ABU0@@Z"
  // M64: define weak_odr dso_local dllexport                noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %struct.SomeStruct* @"??4SomeStruct@@QEAAAEAU0@AEBU0@@Z"
  _Atomic(int) mData;
};
