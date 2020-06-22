// Check the vtable layout for classes with key functions defined in different
// translation units. This TU manifests the vtable for B.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -O1 -S -o - -emit-llvm -fexperimental-relative-c++-abi-vtables | FileCheck %s

#include "cross-tu-header.h"

// CHECK: $_ZN1B3fooEv.stub = comdat any
// CHECK: $_ZN1A3barEv.stub = comdat any
// CHECK: $_ZTI1B.rtti_proxy = comdat any

// CHECK: @_ZTV1B.local = private unnamed_addr constant { [4 x i32] } { [4 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, i8*, i8* }** @_ZTI1B.rtti_proxy to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32] }, { [4 x i32] }* @_ZTV1B.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.B*)* @_ZN1B3fooEv.stub to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32] }, { [4 x i32] }* @_ZTV1B.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.A*)* @_ZN1A3barEv.stub to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32] }, { [4 x i32] }* @_ZTV1B.local, i32 0, i32 0, i32 2) to i64)) to i32)] }, align 4
// CHECK: @_ZTV1B = unnamed_addr alias { [4 x i32] }, { [4 x i32] }* @_ZTV1B.local

// A::bar() is defined outside of the module that defines the vtable for A
// CHECK:      define void @_ZN1A3barEv(%class.A* nocapture %this) unnamed_addr
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

// CHECK:      define void @_ZN1B3fooEv(%class.B* nocapture %this) unnamed_addr
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

// The stubs for B::foo() and A::bar() are hidden
// CHECK:      define hidden void @_ZN1B3fooEv.stub(%class.B* nocapture %0) unnamed_addr #{{[0-9]+}} comdat
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

// CHECK:      define hidden void @_ZN1A3barEv.stub(%class.A* nocapture %0) unnamed_addr #{{[0-9]+}} comdat
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

void A::bar() {}
void B::foo() {}
