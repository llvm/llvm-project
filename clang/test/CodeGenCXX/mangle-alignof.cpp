// RUN: %clang_cc1 -std=c++11 -Wno-gnu-alignof-expression -emit-llvm %s -o - -triple=%itanium_abi_triple | FileCheck %s --check-prefix=CHECK-NEW
// RUN: %clang_cc1 -std=c++11 -Wno-gnu-alignof-expression -emit-llvm %s -o - -triple=%itanium_abi_triple -fclang-abi-compat=11 | FileCheck %s --check-prefix=CHECK-OLD

// Verify the difference in mangling for alignof and __alignof__ in a new ABI
// compat mode.

template <class T> void f1(decltype(alignof(T))) {}
template void f1<int>(__SIZE_TYPE__);
// CHECK-OLD: void @_Z2f1IiEvDTatT_E
// CHECK-NEW: void @_Z2f1IiEvDTatT_E

template <class T> void f2(decltype(__alignof__(T))) {}
template void f2<int>(__SIZE_TYPE__);
// CHECK-OLD: void @_Z2f2IiEvDTatT_E
// CHECK-NEW: void @_Z2f2IiEvDTu11__alignof__T_E

template <class T> void f3(decltype(alignof(T(0)))) {}
template void f3<int>(__SIZE_TYPE__);
// CHECK-OLD: void @_Z2f3IiEvDTazcvT_Li0EE
// CHECK-NEW: void @_Z2f3IiEvDTazcvT_Li0EE

template <class T> void f4(decltype(__alignof__(T(0)))) {}
template void f4<int>(__SIZE_TYPE__);
// CHECK-OLD: void @_Z2f4IiEvDTazcvT_Li0EE
// CHECK-NEW: void @_Z2f4IiEvDTu11__alignof__XcvT_Li0EEEE
