; REQUIRES: x86
; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: llvm-as a.ll -o a.bc
; RUN: llvm-as %p/Inputs/internalize-exportdyn.ll -o b.bc
; RUN: llvm-mc -filetype=obj -triple=x86_64 lib.s -o lib.o
; RUN: ld.lld a.bc b.bc lib.o -o out --export-dynamic -save-temps
; RUN: llvm-dis < out.0.2.internalize.bc | FileCheck %s
; RUN: ld.lld a.bc b.bc lib.o -o out2 -shared -save-temps
; RUN: llvm-dis < out2.0.2.internalize.bc | FileCheck %s --check-prefix=DSO

;--- a.ll
target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

$ext_and_ext = comdat any
$lo_and_ext = comdat any
$lo_and_wo = comdat any
$wo_and_lo = comdat any

@c = linkonce_odr constant i32 1
@g = linkonce_odr global i32 1
@u_c = linkonce_odr unnamed_addr constant i32 1
@u_g = linkonce_odr unnamed_addr global i32 1
@lu_c = linkonce_odr local_unnamed_addr constant i32 1
@lu_g = linkonce_odr local_unnamed_addr global i32 1

declare void @lib(i64)

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

;; Test comdat symbols that are prevailing in this module and non-prevailing in the other module.
define void @ext_and_ext() local_unnamed_addr comdat {
  call void @foo(i64 1)
  ret void
}

;; linkonce_odr in this module and external in the other module.
define linkonce_odr void @lo_and_ext() local_unnamed_addr comdat {
  call void @foo(i64 1)
  ret void
}

;; linkonce_odr in this module and weak_odr in the other module.
define linkonce_odr void @lo_and_wo() local_unnamed_addr comdat {
  ret void
}

define weak_odr void @wo_and_lo() local_unnamed_addr comdat {
  ret void
}

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
; CHECK: define dso_local void @ext_and_ext() comdat
; CHECK-NEXT: call void @foo(i64 1)
; CHECK: define internal void @lo_and_ext() comdat
; CHECK-NEXT: call void @foo(i64 1)
; CHECK: define weak_odr dso_local void @lo_and_wo() comdat
; CHECK: define weak_odr dso_local void @wo_and_lo() comdat

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
; DSO: define void @ext_and_ext() comdat
; DSO: define internal void @lo_and_ext() comdat
; DSO: define weak_odr void @lo_and_wo() comdat
; DSO: define weak_odr void @wo_and_lo() comdat

;--- lib.s
.globl lib
lib:
