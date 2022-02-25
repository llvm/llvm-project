; REQUIRES: x86
; RUN: rm -rf %t && split-file %s %t
; RUN: llvm-as %t/a.ll -o %t/a.bc
; RUN: llvm-mc --triple=x86_64-unknown-linux-gnu -filetype=obj %t/b.s -o %t/b.o
; RUN: llvm-as %t/c.ll -o %t/c.bc
; RUN: not ld.lld %t/a.bc %t/a.bc -o /dev/null -shared 2>&1 | FileCheck %s

;; --thinlto-index-only skips some passes. Test the error is present.
; RUN: not ld.lld %t/a.bc %t/a.bc --thinlto-index-only -o /dev/null 2>&1 | FileCheck %s
; RUN: not ld.lld %t/b.o %t/a.bc --lto-emit-asm -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK2
; RUN: not ld.lld %t/a.bc %t/b.o --thinlto-index-only -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK2

;; --undefined-glob g extracts %t/c.bc which causes a duplicate symbol error.
; RUN: not ld.lld %t/a.bc --start-lib %t/c.bc --undefined-glob g --thinlto-index-only -o /dev/null 2>&1 | FileCheck %s

; CHECK:      duplicate symbol: f
; CHECK-NEXT: >>> defined in {{.*}}.bc
; CHECK-NEXT: >>> defined in {{.*}}.bc

; CHECK2:      duplicate symbol: f
; CHECK2-NEXT: >>> defined in {{.*}}
; CHECK2-NEXT: >>> defined in {{.*}}

;--- a.ll
target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @_start() {
  ret void
}

define void @f() {
  ret void
}

;--- b.s
.globl f
f:

;--- c.ll
target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @f() {
  ret void
}

define void @g() {
  ret void
}
