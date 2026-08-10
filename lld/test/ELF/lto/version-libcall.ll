; REQUIRES: x86
;; The LTO code generator may create references which will fetch lazy symbols.
;; Test that version script local: directives can change the binding of such
;; symbols to STB_LOCAL. This is a bit complex because the LTO code generator
;; happens after version script scanning and can change symbols from Lazy to Defined.

; RUN: llvm-as %s -o %t.bc
; RUN: echo '.globl __udivti3; __udivti3:' | llvm-mc -filetype=obj -triple=x86_64 - -o %t1.o

;; An exact pattern can localize a libcall.
; RUN: echo '{ global: foo; local: __udivti3; };' > %t.exact.ver
; RUN: ld.lld -shared --version-script %t.exact.ver %t.bc --start-lib %t1.o --end-lib -o %t.exact.so
; RUN: llvm-nm %t.exact.so | FileCheck %s

;; A wildcard pattern can localize a libcall.
; RUN: echo '{ global: foo; local: *; };' > %t.wild.ver
; RUN: ld.lld -shared --version-script %t.wild.ver %t.bc --start-lib %t1.o --end-lib -o %t.wild.so
; RUN: llvm-nm %t.wild.so | FileCheck %s

; CHECK: t __udivti3
; CHECK: T foo

;; Test that --dynamic-list works on such libcall fetched symbols.
; RUN: echo '{ foo; __udivti3; };' > %t.exact.list
; RUN: ld.lld -pie --dynamic-list %t.exact.list %t.bc --start-lib %t1.o --end-lib -o %t.exact
; RUN: llvm-nm %t.exact | FileCheck --check-prefix=LIST %s
; RUN: echo '{ foo; __udiv*; };' > %t.wild.list
; RUN: ld.lld -pie --dynamic-list %t.wild.list %t.bc --start-lib %t1.o --end-lib -o %t.wild
; RUN: llvm-nm %t.wild | FileCheck --check-prefix=LIST %s

; LIST: T __udivti3
; LIST: T foo

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i64 @llvm.udiv.fix.i64(i64, i64, i32)

;; The symbol table does not record __udivti3, but the reference will be created
;; on the fly.
define i64 @foo(i64 %x, i64 %y) {
  %ret = call i64 @llvm.udiv.fix.i64(i64 %x, i64 %y, i32 31)
  ret i64 %ret
}
