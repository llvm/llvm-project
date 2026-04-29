; REQUIRES: x86

;; Test that LTO doesn't crash when an alias's target (D2) is in a comdat group
;; that gets dropped because another member (D0) is non-prevailing, while the
;; alias (D1) itself is not in the comdat and survives. With -export-dynamic,
;; the alias can't be internalized away, exposing the dangling reference.
;;
;; This reproduces https://github.com/llvm/llvm-project/issues/190737
;; where -flto -rdynamic on SPEC2006 447.dealII caused:
;;   "Alias must point to a definition"
;;
;; The bug: TU1 puts D0+D2 in comdat $D5 (unified destructor comdat).
;; TU2 puts D0 in its own comdat $D0 (separate, from defaulted destructor).
;; D1 is an alias to D2, not in any comdat.
;; When LTO picks TU2's D0 as prevailing, comdat $D5 is marked non-prevailing.
;; handleNonPrevailingComdat converts D2 to available_externally, but D1
;; (the alias) survives. D2's body is later dropped, leaving D1 pointing
;; to a declaration.

; RUN: split-file %s %t
; RUN: llvm-as %t/tu1.ll -o %t/tu1.o
; RUN: llvm-as %t/tu2.ll -o %t/tu2.o
; RUN: ld.lld -pie -export-dynamic %t/tu2.o %t/tu1.o -o %t/out
; RUN: llvm-nm %t/out | FileCheck %s

; CHECK: W _ZN7DerivedD1Ev

;--- tu1.ll
;; TU with real destructor body: D0 and D2 in same comdat, D1 aliases D2.
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$_ZN7DerivedD5Ev = comdat any

@_ZN7DerivedD1Ev = weak_odr dso_local unnamed_addr alias void (ptr), ptr @_ZN7DerivedD2Ev

define weak_odr dso_local void @_ZN7DerivedD2Ev(ptr %this) unnamed_addr comdat($_ZN7DerivedD5Ev) {
  ret void
}

define weak_odr dso_local void @_ZN7DerivedD0Ev(ptr %this) unnamed_addr comdat($_ZN7DerivedD5Ev) {
  call void @_ZN7DerivedD1Ev(ptr %this)
  ret void
}

;--- tu2.ll
;; TU with defaulted destructor: D0 in its own separate comdat.
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$_ZN7DerivedD0Ev = comdat any

define linkonce_odr dso_local void @_ZN7DerivedD0Ev(ptr %this) unnamed_addr comdat {
  ret void
}

define dso_local void @_start() {
  %p = alloca ptr
  %obj = load ptr, ptr %p
  call void @_ZN7DerivedD0Ev(ptr %obj)
  ret void
}
