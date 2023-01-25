; REQUIRES: x86
;; Test mixed D0/D2 and D5 COMDATs. The file matches llvm/test/ThinLTO/X86/ctor-dtor-alias2.ll

; RUN: rm -rf %t && split-file %s %t && cd %t

;; a.bc defines D0 in comdat D0 and D2 in comdat D2. b.bc defines D0/D1/D2 in comdat D5.
; RUN: opt -module-summary a.ll -o a.bc
; RUN: opt -module-summary b.ll -o b.bc
; RUN: ld.lld -shared a.bc b.bc -o out.so
; RUN: llvm-nm -D out.so

;; Although D0/D2 in b.bc is non-prevailing, keep D1/D2 as definitions, otherwise
;; the output may have an undefined and unsatisfied D1.
; CHECK:      W _ZN1AIiED0Ev
; CHECK-NEXT: W _ZN1AIiED1Ev
; CHECK-NEXT: W _ZN1AIiED2Ev
; CHECK-NEXT: U _ZdlPv
; CHECK-NEXT: T aa
; CHECK-NEXT: T bb

;--- a.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$_ZN1AIiED2Ev = comdat any

$_ZN1AIiED0Ev = comdat any

define void @aa() {
entry:
  %a = alloca ptr, align 8
  call void @_ZN1AIiED2Ev(ptr noundef nonnull %a)
  ret void
}

define linkonce_odr void @_ZN1AIiED2Ev(ptr noundef nonnull %this) unnamed_addr comdat {
  ret void
}

define linkonce_odr void @_ZN1AIiED0Ev(ptr noundef nonnull %this) unnamed_addr comdat {
entry:
  call void @_ZN1AIiED2Ev(ptr noundef nonnull %this)
  call void @_ZdlPv(ptr noundef %this)
  ret void
}

declare void @_ZdlPv(ptr noundef)

;--- b.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$_ZN1AIiED5Ev = comdat any

$_ZTV1AIiE = comdat any

@_ZN1AIiED1Ev = weak_odr unnamed_addr alias void (ptr), ptr @_ZN1AIiED2Ev

define weak_odr void @_ZN1AIiED2Ev(ptr noundef nonnull %this) unnamed_addr comdat($_ZN1AIiED5Ev) {
  ret void
}

define weak_odr void @_ZN1AIiED0Ev(ptr noundef nonnull %this) unnamed_addr comdat($_ZN1AIiED5Ev) {
entry:
  call void @_ZN1AIiED1Ev(ptr noundef nonnull %this)
  call void @_ZdlPv(ptr noundef %this)
  ret void
}

declare void @_ZdlPv(ptr noundef)

define void @bb(ptr noundef %a) {
entry:
  call void @_ZN1AIiED1Ev(ptr noundef nonnull %a)
  call void @_ZdlPv(ptr noundef %a)
  ret void
}
