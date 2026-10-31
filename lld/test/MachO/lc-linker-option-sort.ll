; REQUIRES: x86
; RUN: rm -rf %t; split-file %s %t

; RUN: llc -filetype=obj %t/root-a.ll -o %t/root-a.o
; RUN: llc -filetype=obj %t/root-b.ll -o %t/root-b.o

; RUN: llc -filetype=obj %t/alpha-framework.ll -o %t/alpha-framework.o
; RUN: mkdir -p %t/Alpha.framework
; RUN: llvm-ar rcs %t/Alpha.framework/Alpha %t/alpha-framework.o

; RUN: llc -filetype=obj %t/zed-framework.ll -o %t/zed-framework.o
; RUN: mkdir -p %t/Zed.framework
; RUN: llvm-ar rcs %t/Zed.framework/Zed %t/zed-framework.o

; RUN: llc -filetype=obj %t/mid-framework.ll -o %t/mid-framework.o
; RUN: mkdir -p %t/Mid.framework
; RUN: llvm-ar rcs %t/Mid.framework/Mid %t/mid-framework.o

; RUN: llc -filetype=obj %t/alib.ll -o %t/alib.o
; RUN: llvm-ar rcs %t/libalib.a %t/alib.o

; RUN: llc -filetype=obj %t/zlib.ll -o %t/zlib.o
; RUN: llvm-ar rcs %t/libzlib.a %t/zlib.o

; RUN: llc -filetype=obj %t/mlib.ll -o %t/mlib.o
; RUN: llvm-ar rcs %t/libmlib.a %t/mlib.o

; RUN: %lld -dylib -lSystem -F%t -L%t %t/root-a.o %t/root-b.o \
; RUN:   -map %t/map -o %t/out
; RUN: FileCheck %s < %t/map

;; LC_LINKER_OPTIONs are collected across object files, bucketed by kind, and
;; processed as sorted batches. Frameworks are processed before libraries.
; CHECK-LABEL: # Object files:
; CHECK: root-a.o
; CHECK: root-b.o
; CHECK: Alpha(alpha-framework.o)
; CHECK: Mid(mid-framework.o)
; CHECK: Zed(zed-framework.o)
; CHECK: libalib.a(alib.o)
; CHECK: libmlib.a(mlib.o)
; CHECK: libzlib.a(zlib.o)

;--- root-a.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

!0 = !{!"-lzlib"}
!1 = !{!"-framework", !"Zed"}
!2 = !{!"-lmlib"}
!llvm.linker.options = !{!0, !1, !2}

define void @root_a() {
  call void @zed_framework()
  call void @zlib()
  call void @mlib()
  ret void
}

declare void @zed_framework()
declare void @zlib()
declare void @mlib()

;--- root-b.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

!0 = !{!"-framework", !"Alpha"}
!1 = !{!"-lalib"}
!2 = !{!"-framework", !"Mid"}
!llvm.linker.options = !{!0, !1, !2}

define void @root_b() {
  call void @alpha_framework()
  call void @alib()
  call void @mid_framework()
  ret void
}

declare void @alpha_framework()
declare void @alib()
declare void @mid_framework()

;--- alpha-framework.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @alpha_framework() {
  ret void
}

;--- zed-framework.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @zed_framework() {
  ret void
}

;--- mid-framework.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @mid_framework() {
  ret void
}

;--- alib.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @alib() {
  ret void
}

;--- zlib.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @zlib() {
  ret void
}

;--- mlib.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @mlib() {
  ret void
}
