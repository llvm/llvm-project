; REQUIRES: x86
; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %p/Inputs/internalize-exportdyn.ll -o %t2.o
; RUN: ld.lld %t.o %t2.o -o %t2 --export-dynamic -save-temps
; RUN: llvm-dis < %t2.0.2.internalize.bc | FileCheck %s
; RUN: ld.lld %t.o %t2.o -o %t3 -shared -save-temps
; RUN: llvm-dis < %t3.0.2.internalize.bc | FileCheck %s --check-prefix=DSO

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

@c = linkonce_odr constant i32 1
@g = linkonce_odr global i32 1
@u_c = linkonce_odr unnamed_addr constant i32 1
@u_g = linkonce_odr unnamed_addr global i32 1
@lu_c = linkonce_odr local_unnamed_addr constant i32 1
@lu_g = linkonce_odr local_unnamed_addr global i32 1

define void @_start() {
  ret void
}

define void @foo() {
  ret void
}

define hidden void @bar() {
  ret void
}

define linkonce_odr void @zed() local_unnamed_addr {
  ret void
}

define linkonce_odr void @zed2() unnamed_addr {
  ret void
}

define linkonce_odr void @bah() {
  ret void
}

define linkonce_odr void @baz() {
  ret void
}

@use_baz = global ptr @baz

; Check what gets internalized.
; CHECK: @c = weak_odr dso_local constant i32 1
; CHECK: @g = weak_odr dso_local global i32 1
; CHECK: @u_c = internal unnamed_addr constant i32 1
; CHECK: @u_g = internal unnamed_addr global i32 1
; CHECK: @lu_c = internal constant i32 1
; CHECK: @lu_g = weak_odr dso_local global i32 1
; CHECK: define dso_local void @_start()
; CHECK: define dso_local void @foo()
; CHECK: define internal void @bar()
; CHECK: define internal void @zed()
; CHECK: define internal void @zed2()
; CHECK: define weak_odr dso_local void @bah()
; CHECK: define weak_odr dso_local void @baz()

; DSO: @c = weak_odr constant i32 1
; DSO: @g = weak_odr global i32 1
; DSO: @u_c = internal unnamed_addr constant i32 1
; DSO: @u_g = internal unnamed_addr global i32 1
; DSO: @lu_c = internal constant i32 1
; DSO: @lu_g = weak_odr global i32 1
; DSO: define void @_start()
; DSO: define void @foo()
; DSO: define internal void @bar()
; DSO: define internal void @zed()
; DSO: define internal void @zed2()
; DSO: define weak_odr void @bah()
; DSO: define weak_odr void @baz()
