// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple=x86_64-linux-gnu | FileCheck %s

struct S {};

template <class T> void f1(decltype(__builtin_structured_binding_size(T))) {}
template void f1<S>(__SIZE_TYPE__);
// CHECK: void @_Z2f1I1SEvDTu11__builtin_structured_binding_sizeT_EE

template <class T> void f2(decltype(__builtin_structured_binding_size(T{}))) {}
template void f2<S>(__SIZE_TYPE__);
// CHECK: void @_Z2f2I1SEvDTu11__builtin_structured_binding_sizeXtlT_EEEE

