// RUN: %clang_cc1 -no-opaque-pointers %s -fms-extensions -triple x86_64-windows-msvc -emit-llvm -o - | FileCheck %s

template <typename> struct MessageT { };
extern template struct MessageT<int>;

// CHECK: define weak_odr dso_local dllexport {{.*}} %struct.MessageT* @"??4?$MessageT@H@@QEAAAEAU0@AEBU0@@Z"(
template struct __declspec(dllexport) MessageT<int>;
// Previously we crashed when this dllexport was the last thing in the file.
// DO NOT ADD MORE TESTS AFTER THIS LINE!
