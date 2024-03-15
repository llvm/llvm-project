; UNSUPPORTED: system-windows
; REQUIRES: shell, non-root-user
; RUN: llvm-as %s -o %t.bc
; RUN: touch %t.resolution.txt
; RUN: chmod u-w %t.resolution.txt
; RUN: not ld.lld -save-temps %t.bc -o %t 2>&1 | FileCheck -DMSG=%errc_EACCES %s
; RUN: rm -f %t.resolution.txt

; CHECK: error: [[MSG]]

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @_start() {
  ret void
}
