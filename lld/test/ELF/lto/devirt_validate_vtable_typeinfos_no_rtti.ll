; REQUIRES: x86

;; Common artifacts
; RUN: opt --thinlto-bc -o %t1.o %s
; RUN: opt --thinlto-bc --thinlto-split-lto-unit -o %t1_hybrid.o %s
; RUN: cat %s > %t1_regular.ll
; RUN: echo '!llvm.module.flags = !{!6, !7}' >> %t1_regular.ll
; RUN: echo '!6 = !{i32 1, !"ThinLTO", i32 0}' >> %t1_regular.ll
; RUN: echo '!7 = !{i32 1, !"EnableSplitLTOUnit", i32 1}' >> %t1_regular.ll
; RUN: opt -module-summary -o %t1_regular.o %t1_regular.ll

;; With --lto-whole-program-visibility, we assume no native types can interfere
;; and thus proceed with devirtualization even in the presence of native types

;; Index based WPD
; RUN: ld.lld %t1.o -o %t3_index -save-temps --lto-whole-program-visibility \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t1.o.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-COMMON-IR-LABEL,CHECK-IR

;; Hybrid WPD
; RUN: ld.lld %t1_hybrid.o -o %t3_hybrid -save-temps --lto-whole-program-visibility \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t1_hybrid.o.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-COMMON-IR-LABEL,CHECK-IR

;; Regular LTO WPD
; RUN: ld.lld %t1_regular.o -o %t3_regular -save-temps --lto-whole-program-visibility \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t3_regular.0.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-COMMON-IR-LABEL,CHECK-IR

; REMARK-DAG: single-impl: devirtualized a call to _ZN1A1nEi
; REMARK-DAG: single-impl: devirtualized a call to _ZN1D1mEi

;; With --lto-whole-program-visibility and --lto-validate-all-vtables-have-type-infos
;; we rely on resolutions on the typename symbol to inform us of what's outside the summary.
;; Without the typename symbol in the LTO unit (e.g. RTTI disabled) this causes
;; conservative disablement of WPD on these types unless it's local

;; Index based WPD
; RUN: ld.lld %t1.o -o %t3_index -save-temps --lto-whole-program-visibility --lto-validate-all-vtables-have-type-infos \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=VALIDATE
; RUN: llvm-dis %t1.o.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-COMMON-IR-LABEL,CHECK-VALIDATE-IR

;; Hybrid WPD
; RUN: ld.lld %t1_hybrid.o -o %t3_hybrid -save-temps --lto-whole-program-visibility --lto-validate-all-vtables-have-type-infos \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=VALIDATE
; RUN: llvm-dis %t1_hybrid.o.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-COMMON-IR-LABEL,CHECK-VALIDATE-IR

;; Regular LTO WPD
; RUN: ld.lld %t1_regular.o -o %t3_regular -save-temps --lto-whole-program-visibility --lto-validate-all-vtables-have-type-infos \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=VALIDATE
; RUN: llvm-dis %t3_regular.0.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-COMMON-IR-LABEL,CHECK-VALIDATE-IR

; VALIDATE-DAG: single-impl: devirtualized a call to _ZN1D1mEi

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.A = type { ptr }
%struct.B = type { %struct.A }
%struct.C = type { %struct.A }
%struct.D = type { ptr }

@_ZTV1B = linkonce_odr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr null, ptr @_ZN1B1fEi, ptr @_ZN1A1nEi] }, !type !0, !type !1
@_ZTV1C = linkonce_odr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr null, ptr @_ZN1C1fEi, ptr @_ZN1A1nEi] }, !type !0, !type !2
@_ZTV1D = internal constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN1D1mEi] }, !type !3, !vcall_visibility !5

;; Prevent the vtables from being dead code eliminated.
@llvm.used = appending global [3 x ptr] [ ptr @_ZTV1B, ptr @_ZTV1C, ptr @_ZTV1D ]

; CHECK-COMMON-IR-LABEL: define dso_local {{(noundef )?}}i32 @_start
define i32 @_start(ptr %obj, ptr %obj2, i32 %a) {
entry:
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"_ZTS1A")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr ptr, ptr %vtable, i32 1
  %fptr1 = load ptr, ptr %fptrptr, align 8

  ;; Check that the call was devirtualized.
  ; CHECK-IR: %call = tail call i32 @_ZN1A1nEi
  ;; No resolution for _ZTS1A means we don't devirtualize
  ; CHECK-VALIDATE-IR: %call = tail call i32 %fptr1
  %call = tail call i32 %fptr1(ptr nonnull %obj, i32 %a)

  %fptr22 = load ptr, ptr %vtable, align 8

  ;; We still have to call it as virtual.
  ; CHECK-IR: %call3 = tail call i32 %fptr22
  ; CHECK-VALIDATE-IR: %call3 = tail call i32 %fptr22
  %call3 = tail call i32 %fptr22(ptr nonnull %obj, i32 %call)

  %vtable2 = load ptr, ptr %obj2
  %p2 = call i1 @llvm.type.test(ptr %vtable2, metadata !4)
  call void @llvm.assume(i1 %p2)

  %fptr33 = load ptr, ptr %vtable2, align 8

  ;; Check that the call was devirtualized.
  ; CHECK-IR: %call4 = tail call i32 @_ZN1D1mEi
  ;; Being local this has VCallVisibilityTranslationUnit
  ;; visibility so it's still devirtualized
  ; CHECK-VALIDATE-IR: %call4 = tail call i32 @_ZN1D1mEi
  %call4 = tail call i32 %fptr33(ptr nonnull %obj2, i32 %call3)
  ret i32 %call4
}
; CHECK-COMMON-IR-LABEL: ret i32
; CHECK-COMMON-IR-LABEL: }

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

define linkonce_odr i32 @_ZN1B1fEi(ptr %this, i32 %a) #0 {
   ret i32 0;
}

define linkonce_odr i32 @_ZN1A1nEi(ptr %this, i32 %a) #0 {
   ret i32 0;
}

define linkonce_odr i32 @_ZN1C1fEi(ptr %this, i32 %a) #0 {
   ret i32 0;
}

define internal i32 @_ZN1D1mEi(ptr %this, i32 %a) #0 {
   ret i32 0;
}

;; Make sure we don't inline or otherwise optimize out the direct calls.
attributes #0 = { noinline optnone }

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTS1B"}
!2 = !{i64 16, !"_ZTS1C"}
!3 = !{i64 16, !4}
!4 = distinct !{}
!5 = !{i64 2}
