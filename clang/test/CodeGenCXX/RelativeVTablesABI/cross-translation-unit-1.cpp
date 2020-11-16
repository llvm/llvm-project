// Check the vtable layout for classes with key functions defined in different
// translation units. This TU only manifests the vtable for A.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -O1 -S -o - -emit-llvm -fexperimental-relative-c++-abi-vtables | FileCheck %s

#include "cross-tu-header.h"

// CHECK: $_ZN1A3fooEv.stub = comdat any
// CHECK: $_ZN1A3barEv.stub = comdat any
// CHECK: $_ZTI1A.rtti_proxy = comdat any

// CHECK: @_ZTV1A.local = private unnamed_addr constant { [4 x i32] } { [4 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, i8* }** @_ZTI1A.rtti_proxy to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32] }, { [4 x i32] }* @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.A*)* @_ZN1A3fooEv.stub to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32] }, { [4 x i32] }* @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.A*)* @_ZN1A3barEv.stub to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [4 x i32] }, { [4 x i32] }* @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32)] }, align 4
// @_ZTV1A = unnamed_addr alias { [4 x i32] }, { [4 x i32] }* @_ZTV1A.local

// A::foo() is still available for other modules to use since it is not marked with private or internal linkage.
// CHECK:      define void @_ZN1A3fooEv(%class.A* nocapture {{[^,]*}} %this) unnamed_addr
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

// The proxy that we take a reference to in the vtable has hidden visibility and external linkage so it can be used only by other modules in the same DSO. A::foo() is inlined into this stub since it is defined in the same module.
// CHECK:      define hidden void @_ZN1A3fooEv.stub(%class.A* nocapture {{[^,]*}} %0) unnamed_addr #{{[0-9]+}} comdat
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

// A::bar() is called within the module but not defined, even though the VTable for A is emitted here
// CHECK: declare void @_ZN1A3barEv(%class.A* {{[^,]*}}) unnamed_addr

// The stub for A::bar() is made private, so it will not appear in the symbol table and is only used in this module. We tail call here because A::bar() is not defined in the same module.
// CHECK:      define hidden void @_ZN1A3barEv.stub(%class.A* {{[^,]*}} %0) unnamed_addr {{#[0-9]+}} comdat {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN1A3barEv(%class.A* {{[^,]*}} %0)
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

void A::foo() {}
void A_foo(A *a) { a->foo(); }
void A_bar(A *a) { a->bar(); }
