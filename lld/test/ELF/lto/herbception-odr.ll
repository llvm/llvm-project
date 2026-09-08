; REQUIRES: x86
;; Herbception ("throws"/"fails{E}") ODR checking during LTO: the herbception
;; specifier is not part of the mangled name, so translation units that
;; disagree about it produce identical symbol names with incompatible ABIs.
;; LTO must diagnose this.

; RUN: llvm-as %p/Inputs/herbception-odr-throws.ll -o %tthrows.o
; RUN: llvm-as %p/Inputs/herbception-odr-other-payload.ll -o %tother.o
; RUN: llvm-as %p/Inputs/herbception-odr-plain-decl.ll -o %tdecl.o

;; A throws definition linked with a plain declaration of the same symbol is
;; an ODR violation (the plain caller uses the wrong ABI).
; RUN: not ld.lld %tthrows.o %tdecl.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=DECL

; DECL: error: herbception ODR violation: symbol 'f' has conflicting definitions: it is defined as a herbception ('throws') function with error payload type '{ ptr, i64 }' in '{{.*}}throws.o', but without the herbception error channel in '{{.*}}decl.o'

;; Two throws definitions with different error payloads also conflict.
; RUN: not ld.lld %tthrows.o %tother.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=PAYLOAD

; PAYLOAD: error: herbception ODR violation: symbol 'f' has conflicting definitions: it is defined with herbception error payload type '{ ptr, i64 }' in '{{.*}}throws.o', but with payload type 'i64' in '{{.*}}other.o'

;; Consistent signatures link fine.
; RUN: llvm-as %s -o %tstart.o
; RUN: ld.lld --lto-O0 %tthrows.o %tstart.o -o %tok
; RUN: llvm-nm %tok | FileCheck %s --check-prefix=OK

; OK: T _start
; OK: t f

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @_start() {
entry:
  %r = call { { ptr, i64 }, i1 } @f()
  ret void
}

declare { { ptr, i64 }, i1 } @f() #0

attributes #0 = { throws }
