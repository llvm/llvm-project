; Overriding the module triple must not keep a DataLayout computed for the old
; triple. Mach-O x86_64 uses "m:o"; the input is ELF ("m:e").

; REQUIRES: x86-registered-target
; RUN: %clang_cc1 -triple x86_64-apple-macosx -x ir %s -emit-obj -o /dev/null 2>&1 \
; RUN:   | FileCheck --check-prefix=WARN %s
; RUN: %clang_cc1 -triple x86_64-apple-macosx -x ir %s -emit-llvm -o - 2>/dev/null \
; RUN:   | FileCheck %s

; WARN: warning: overriding the module target triple with x86_64-apple-macosx

; CHECK: target datalayout = "e-m:o-
; CHECK: target triple = "x86_64-apple-macosx"

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @f() {
  ret i32 0
}
