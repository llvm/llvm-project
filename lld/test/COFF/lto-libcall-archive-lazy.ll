; REQUIRES: x86
; RUN: rm -rf %t && split-file %s %t

; An archive member that is bitcode and defines a library function the code
; generator may call is only added to the link when code generation does call
; it, and then LTO is run again with it. A member that defines a function that
; is not called is not added, so its undefined references do not matter.

; RUN: llvm-as -o %t/main.obj %t/main.ll
; RUN: llvm-as -o %t/memcpy.obj %t/memcpy.ll
; RUN: llvm-as -o %t/fmaf.obj %t/fmaf.ll
; RUN: llvm-ar rcs %t/lib.a %t/memcpy.obj %t/fmaf.obj

; RUN: lld-link -out:%t/used.exe -subsystem:console -entry:used -safeseh:no -lldmap:- %t/main.obj %t/lib.a | FileCheck %s --check-prefix=USED
; USED: _used
; USED: _memcpy
; USED-NOT: _unused_dependency

; RUN: lld-link -out:%t/unused.exe -subsystem:console -entry:unused -safeseh:no -lldmap:- %t/main.obj %t/lib.a | FileCheck %s --check-prefix=UNUSED
; UNUSED: _unused
; UNUSED-NOT: _memcpy

;--- main.ll
target datalayout = "e-m:x-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-unknown-windows"

define void @used(ptr %a, ptr %b) {
  call void @llvm.memcpy.p0.p0.i64(ptr %a, ptr %b, i64 1024, i1 false)
  ret void
}

define void @unused() {
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1)

;--- memcpy.ll
target datalayout = "e-m:x-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-unknown-windows"

define void @memcpy() {
  ret void
}

;--- fmaf.ll
target datalayout = "e-m:x-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-unknown-windows"

declare void @unused_dependency()

define float @fmaf(float %a, float %b, float %c) {
  call void @unused_dependency()
  ret float %a
}
