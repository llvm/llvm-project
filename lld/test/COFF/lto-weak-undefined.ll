; REQUIRES: x86

;; Test linking of weak symbols with LTO. The weak symbol may be defined
;; by another object file, or may be left undefined. When compiling the
;; IR with an undefined weak symbol, the emitted object file will contain
;; a weak alias pointing at an absolute symbol for the address null.
;; Make sure both cases can be linked correctly.

; RUN: split-file %s %t.dir
; RUN: llvm-as %t.dir/main.ll -o %t.main.obj
; RUN: llvm-as %t.dir/optional.ll -o %t.optional.obj

; RUN: lld-link /entry:main %t.main.obj /out:%t-undef.exe
; RUN: lld-link /entry:main %t.main.obj %t.optional.obj /out:%t-def.exe

;--- main.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define dso_local i32 @main() {
entry:
  br i1 icmp ne (ptr @optionalFunc, ptr null), label %if.then, label %if.end

if.then:
  tail call void @optionalFunc()
  br label %if.end

if.end:
  ret i32 0
}

declare extern_weak void @optionalFunc()

;--- optional.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define void @optionalFunc() {
  ret void
}
