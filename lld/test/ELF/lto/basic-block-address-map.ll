; REQUIRES: x86
; RUN: llvm-as %s -o %t.o
; RUN: ld.lld %t.o -o %t --lto-basic-block-address-map --lto-O0 --save-temps
; RUN: llvm-readobj --sections %t.lto.o | FileCheck --check-prefix=SECNAMES %s

; SECNAMES: Type: SHT_LLVM_BB_ADDR_MAP

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define dso_local void @foo(i32 %b) local_unnamed_addr {
entry:
  %tobool.not = icmp eq i32 %b, 0
  br i1 %tobool.not, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  tail call void @foo(i32 0)
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  ret void
}

define void @_start() {
  call void @foo(i32 1)
  ret void
}
