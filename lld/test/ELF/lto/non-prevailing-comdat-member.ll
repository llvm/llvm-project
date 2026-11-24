; REQUIRES: x86
; Test that a non-prevailing D0 in a D5 COMDAT does not discard D1 and D2,
; in both pure Regular LTO and mixed ThinLTO + Regular LTO links.
; RUN: split-file %s %t
; RUN: llvm-as %t/a.ll -o %t/a.bc
; RUN: llvm-as %t/b.ll -o %t/b.bc
; RUN: ld.lld -shared %t/a.bc %t/b.bc -o %t/forward.so
; RUN: llvm-nm -D %t/forward.so | FileCheck %s
; RUN: ld.lld -shared %t/b.bc %t/a.bc -o %t/reverse.so
; RUN: llvm-nm -D %t/reverse.so | FileCheck %s
; RUN: opt -module-summary %t/a.ll -o %t/a.thin.bc
; RUN: ld.lld -shared %t/a.thin.bc %t/b.bc -o %t/mixed-forward.so
; RUN: llvm-nm -D %t/mixed-forward.so | FileCheck %s
; RUN: ld.lld -shared %t/b.bc %t/a.thin.bc -o %t/mixed-reverse.so
; RUN: llvm-nm -D %t/mixed-reverse.so | FileCheck %s
;
; CHECK: W _ZN1AIiED0Ev
; CHECK-NEXT: W _ZN1AIiED1Ev
; CHECK-NEXT: W _ZN1AIiED2Ev

;--- a.ll
target datalayout = "e-m:e-p270:32:32:32-p271:32:32:32-p272:64:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$_ZN1AIiED0Ev = comdat any

define linkonce_odr void @_ZN1AIiED0Ev(ptr %this) unnamed_addr comdat {
  ret void
}

;--- b.ll
target datalayout = "e-m:e-p270:32:32:32-p271:32:32:32-p272:64:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$_ZN1AIiED5Ev = comdat any

@_ZN1AIiED1Ev = weak_odr unnamed_addr alias void (ptr), ptr @_ZN1AIiED2Ev

define weak_odr void @_ZN1AIiED2Ev(ptr %this) unnamed_addr comdat($_ZN1AIiED5Ev) {
  ret void
}

define weak_odr void @_ZN1AIiED0Ev(ptr %this) unnamed_addr comdat($_ZN1AIiED5Ev) {
  ret void
}
