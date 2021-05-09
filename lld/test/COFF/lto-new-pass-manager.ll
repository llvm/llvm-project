; REQUIRES: x86
; RUN: llvm-as %s -o %t.obj

; RUN: lld-link %t.obj -entry:main -opt:ltonewpassmanager -opt:ltodebugpassmanager 2>&1 | FileCheck %s --check-prefix=ENABLED
; ENABLED: Running pass: InstCombinePass

; Passing -time just to avoid empty FileCheck input
; RUN: lld-link %t.obj -entry:main -time -opt:ltonewpassmanager -opt:ltodebugpassmanager -opt:noltonewpassmanager 2>&1 | FileCheck %s --check-prefix=DISABLED
; RUN: lld-link %t.obj -entry:main -time -opt:ltonewpassmanager -opt:ltodebugpassmanager -opt:noltodebugpassmanager 2>&1 | FileCheck %s --check-prefix=DISABLED
; DISABLED-NOT: Running pass: InstCombinePass

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.11.0"

define dso_local i32 @main(i32 %argc, i8** nocapture readnone %0) local_unnamed_addr {
entry:
  ret i32 %argc
}
