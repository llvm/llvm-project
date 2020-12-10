// Check the vtable layout for classes with key functions defined in different
// translation units. This TU only manifests the vtable for A.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -O1 -S -o - -emit-llvm -fexperimental-relative-c++-abi-vtables | FileCheck %s

#include "cross-tu-header.h"

// CHECK: $_ZTI1A.rtti_proxy = comdat any

// CHECK: @_ZTV1A.local = private unnamed_addr constant { [4 x i32] } { [4 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, i8* }** @_ZTI1A.rtti_proxy to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32] }, { [4 x i32] }* @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.A*)* dso_local_equivalent @_ZN1A3fooEv to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32] }, { [4 x i32] }* @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.A*)* dso_local_equivalent @_ZN1A3barEv to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32] }, { [4 x i32] }* @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32)] }, align 4
// @_ZTV1A = unnamed_addr alias { [4 x i32] }, { [4 x i32] }* @_ZTV1A.local

void A::foo() {}
void A_foo(A *a) { a->foo(); }
void A_bar(A *a) { a->bar(); }
