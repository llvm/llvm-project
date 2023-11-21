; REQUIRES: x86

;; Common artifacts
; RUN: opt --thinlto-bc -o %t1.o %s
; RUN: opt --thinlto-bc --thinlto-split-lto-unit -o %t1_hybrid.o %s
; RUN: cat %s > %t1_regular.ll
; RUN: echo '!llvm.module.flags = !{!12, !13}' >> %t1_regular.ll
; RUN: echo '!12 = !{i32 1, !"ThinLTO", i32 0}' >> %t1_regular.ll
; RUN: echo '!13 = !{i32 1, !"EnableSplitLTOUnit", i32 1}' >> %t1_regular.ll
; RUN: opt -module-summary -o %t1_regular.o %t1_regular.ll

; RUN: llvm-as %S/Inputs/devirt_validate_vtable_typeinfos.ll -o %t2.bc
; RUN: llc -relocation-model=pic -filetype=obj %t2.bc -o %t2.o
; RUN: ld.lld %t2.o -o %t2.so -shared

; RUN: llvm-as %S/Inputs/devirt_validate_vtable_typeinfos_no_rtti.ll -o %t2_nortti.bc
; RUN: llc -relocation-model=pic -filetype=obj %t2_nortti.bc -o %t2_nortti.o
; RUN: ld.lld %t2_nortti.o -o %t2_nortti.so -shared

; RUN: llvm-as %S/Inputs/devirt_validate_vtable_typeinfos_undef.ll -o %t2_undef.bc
; RUN: llc -relocation-model=pic -filetype=obj %t2_undef.bc -o %t2_undef.o
; RUN: ld.lld %t2_undef.o -o %t2_undef.so -shared

;; With --lto-whole-program-visibility, we assume no native types can interfere
;; and thus proceed with devirtualization even in the presence of native types

;; Index based WPD
; RUN: ld.lld %t1.o %t2.o -o %t3_index -save-temps --lto-whole-program-visibility \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t1.o.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-COMMON-IR-LABEL,CHECK-IR

;; Hybrid WPD
; RUN: ld.lld %t1_hybrid.o %t2.o -o %t3_hybrid -save-temps --lto-whole-program-visibility \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t1_hybrid.o.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-COMMON-IR-LABEL,CHECK-IR

;; Regular LTO WPD
; RUN: ld.lld %t1_regular.o %t2.o -o %t3_regular -save-temps --lto-whole-program-visibility \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t3_regular.0.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-COMMON-IR-LABEL,CHECK-IR

; REMARK-DAG: single-impl: devirtualized a call to _ZN1A1nEi
; REMARK-DAG: single-impl: devirtualized a call to _ZN1D1mEi

;; With --lto-validate-all-vtables-have-type-infos, the linker checks for the presence of vtables
;; and RTTI in native files and blocks devirtualization to be conservative on correctness
;; for these types.

;; Index based WPD
; RUN: ld.lld %t1.o %t2.o -o %t4_index -save-temps --lto-whole-program-visibility --lto-validate-all-vtables-have-type-infos \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=VALIDATE
; RUN: llvm-dis %t1.o.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-COMMON-IR-LABEL,CHECK-VALIDATE-IR

;; Hybrid WPD
; RUN: ld.lld %t1_hybrid.o %t2.o -o %t4_hybrid -save-temps --lto-whole-program-visibility --lto-validate-all-vtables-have-type-infos \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=VALIDATE
; RUN: llvm-dis %t1_hybrid.o.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-COMMON-IR-LABEL,CHECK-VALIDATE-IR

;; Regular LTO WPD
; RUN: ld.lld %t1_regular.o %t2.o -o %t4_regular -save-temps --lto-whole-program-visibility --lto-validate-all-vtables-have-type-infos \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=VALIDATE
; RUN: llvm-dis %t4_regular.0.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-COMMON-IR-LABEL,CHECK-VALIDATE-IR

;; DSOs behave similarly

;; Index based WPD
; RUN: ld.lld %t1.o %t2.so -o %t5_index -save-temps --lto-whole-program-visibility --lto-validate-all-vtables-have-type-infos \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=VALIDATE
; RUN: llvm-dis %t1.o.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-COMMON-IR-LABEL,CHECK-VALIDATE-IR

;; Hybrid WPD
; RUN: ld.lld %t1_hybrid.o %t2.so -o %t5_hybrid -save-temps --lto-whole-program-visibility --lto-validate-all-vtables-have-type-infos \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=VALIDATE
; RUN: llvm-dis %t1_hybrid.o.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-COMMON-IR-LABEL,CHECK-VALIDATE-IR

;; Regular LTO WPD
; RUN: ld.lld %t1_regular.o %t2.so -o %t5_regular -save-temps --lto-whole-program-visibility --lto-validate-all-vtables-have-type-infos \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=VALIDATE
; RUN: llvm-dis %t5_regular.0.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-COMMON-IR-LABEL,CHECK-VALIDATE-IR

; VALIDATE-NOT: single-impl:
; VALIDATE:     single-impl: devirtualized a call to _ZN1D1mEi
; VALIDATE-NOT: single-impl:

;; When vtables without type infos are detected in native files, we have a hole in our knowledge so
;; --lto-validate-all-vtables-have-type-infos conservatively disables --lto-whole-program-visibility

;; Index based WPD
; RUN: ld.lld %t1.o %t2_nortti.o -o %t6_index -save-temps --lto-whole-program-visibility --lto-validate-all-vtables-have-type-infos \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=NO-RTTI
; RUN: llvm-dis %t1.o.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-COMMON-IR-LABEL,CHECK-NO-RTTI-IR

;; Hybrid WPD
; RUN: ld.lld %t1_hybrid.o %t2_nortti.o -o %t6_hybrid -save-temps --lto-whole-program-visibility --lto-validate-all-vtables-have-type-infos \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=NO-RTTI
; RUN: llvm-dis %t1_hybrid.o.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-COMMON-IR-LABEL,CHECK-NO-RTTI-IR

;; Regular LTO WPD
; RUN: ld.lld %t1_regular.o %t2_nortti.o -o %t6_regular -save-temps --lto-whole-program-visibility --lto-validate-all-vtables-have-type-infos \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=NO-RTTI
; RUN: llvm-dis %t6_regular.0.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-COMMON-IR-LABEL,CHECK-NO-RTTI-IR

;; DSOs behave similarly

;; Index based WPD
; RUN: ld.lld %t1.o %t2_nortti.so -o %t7_index -save-temps --lto-whole-program-visibility --lto-validate-all-vtables-have-type-infos \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=NO-RTTI
; RUN: llvm-dis %t1.o.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-COMMON-IR-LABEL,CHECK-NO-RTTI-IR

;; Hybrid WPD
; RUN: ld.lld %t1_hybrid.o %t2_nortti.so -o %t7_hybrid -save-temps --lto-whole-program-visibility --lto-validate-all-vtables-have-type-infos \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=NO-RTTI
; RUN: llvm-dis %t1_hybrid.o.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-COMMON-IR-LABEL,CHECK-NO-RTTI-IR

;; Regular LTO WPD
; RUN: ld.lld %t1_regular.o %t2_nortti.so -o %t7_regular -save-temps --lto-whole-program-visibility --lto-validate-all-vtables-have-type-infos \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=NO-RTTI
; RUN: llvm-dis %t7_regular.0.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-COMMON-IR-LABEL,CHECK-NO-RTTI-IR

; NO-RTTI-DAG: --lto-validate-all-vtables-have-type-infos: RTTI missing for vtable _ZTV6Native, --lto-whole-program-visibility disabled
; NO-RTTI-DAG: single-impl: devirtualized a call to _ZN1D1mEi

;; --lto-known-safe-vtables=* can be used to specifically allow types to participate in WPD
;; even if they don't have corresponding RTTI

;; Index based WPD
; RUN: ld.lld %t1.o %t2_nortti.o -o %t8_index -save-temps --lto-whole-program-visibility --lto-validate-all-vtables-have-type-infos \
; RUN:   --lto-known-safe-vtables=_ZTV6Native -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t1.o.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-COMMON-IR-LABEL,CHECK-IR

;; Hybrid WPD
; RUN: ld.lld %t1_hybrid.o %t2_nortti.o -o %t8_hybrid -save-temps --lto-whole-program-visibility --lto-validate-all-vtables-have-type-infos \
; RUN:   --lto-known-safe-vtables=_ZTV6Native -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t1_hybrid.o.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-COMMON-IR-LABEL,CHECK-IR

;; Regular LTO WPD
; RUN: ld.lld %t1_regular.o %t2_nortti.o -o %t8_regular -save-temps --lto-whole-program-visibility --lto-validate-all-vtables-have-type-infos \
; RUN:   --lto-known-safe-vtables=_ZTV6Native -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t8_regular.0.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-COMMON-IR-LABEL,CHECK-IR

;; Only check for definitions of vtables symbols, just having a reference does not allow a type to
;; be derived from

;; Index based WPD
; RUN: ld.lld %t1.o %t2_undef.o -o %t9_index -save-temps --lto-whole-program-visibility --lto-validate-all-vtables-have-type-infos \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t1.o.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-COMMON-IR-LABEL,CHECK-IR

;; Hybrid WPD
; RUN: ld.lld %t1_hybrid.o %t2_undef.o -o %t9_hybrid -save-temps --lto-whole-program-visibility --lto-validate-all-vtables-have-type-infos \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t1_hybrid.o.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-COMMON-IR-LABEL,CHECK-IR

;; Regular LTO WPD
; RUN: ld.lld %t1_regular.o %t2_undef.o -o %t9_regular -save-temps --lto-whole-program-visibility --lto-validate-all-vtables-have-type-infos \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t9_regular.0.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-COMMON-IR-LABEL,CHECK-IR

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.A = type { ptr }
%struct.B = type { %struct.A }
%struct.C = type { %struct.A }
%struct.D = type { ptr }

@_ZTV1B = linkonce_odr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTI1B, ptr @_ZN1B1fEi, ptr @_ZN1A1nEi] }, !type !0, !type !1, !type !2, !type !3, !type !4, !type !5
@_ZTV1C = linkonce_odr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTI1C, ptr @_ZN1C1fEi, ptr @_ZN1A1nEi] }, !type !0, !type !1, !type !2, !type !6, !type !7, !type !8
@_ZTV1D = internal constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI1D, ptr @_ZN1D1mEi] }, !type !9, !vcall_visibility !11

@_ZTS1A = linkonce_odr constant [3 x i8] c"1A\00"
@_ZTI1A = linkonce_odr constant { ptr, ptr } { ptr null, ptr @_ZTS1A }

@_ZTS1B = linkonce_odr constant [3 x i8] c"1B\00"
@_ZTI1B = linkonce_odr constant { ptr, ptr, ptr } { ptr null, ptr @_ZTS1B, ptr @_ZTI1A }

@_ZTS1C = linkonce_odr constant [3 x i8] c"1C\00"
@_ZTI1C = linkonce_odr constant { ptr, ptr, ptr } { ptr null, ptr @_ZTS1C, ptr @_ZTI1A }

@_ZTS1D = internal constant [3 x i8] c"1D\00"
@_ZTI1D = internal constant { ptr, ptr } { ptr null, ptr @_ZTS1D }

;; Prevent the vtables from being dead code eliminated.
@llvm.used = appending global [3 x ptr] [ ptr @_ZTV1B, ptr @_ZTV1C, ptr @_ZTV1D ]

; CHECK-COMMON-IR-LABEL: define dso_local i32 @_start
define i32 @_start(ptr %obj, ptr %obj2, i32 %a) {
entry:
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"_ZTS1A")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr ptr, ptr %vtable, i32 1
  %fptr1 = load ptr, ptr %fptrptr, align 8

  ;; Check that the call was devirtualized.
  ; CHECK-IR: %call = tail call i32 @_ZN1A1nEi
  ;; --lto-whole-program-visibility disabled so no devirtualization
  ; CHECK-VALIDATE-IR: %call = tail call i32 %fptr1
  ; CHECK-NO-RTTI-IR: %call = tail call i32 %fptr1
  %call = tail call i32 %fptr1(ptr nonnull %obj, i32 %a)

  %fptr22 = load ptr, ptr %vtable, align 8

  ;; We still have to call it as virtual.
  ; CHECK-IR: %call2 = tail call i32 %fptr22
  ; CHECK-VALIDATE-IR: %call2 = tail call i32 %fptr22
  ; CHECK-NO-RTTI-IR: %call2 = tail call i32 %fptr22
  %call2 = tail call i32 %fptr22(ptr nonnull %obj, i32 %call)

  %vtable2 = load ptr, ptr %obj2
  %p2 = call i1 @llvm.type.test(ptr %vtable2, metadata !10)
  call void @llvm.assume(i1 %p2)

  %fptr33 = load ptr, ptr %vtable2, align 8

  ;; Check that the call was devirtualized.
  ; CHECK-IR: %call3 = tail call i32 @_ZN1D1mEi
  ;; Types not present in native files can still be devirtualized
  ; CHECK-VALIDATE-IR: %call3 = tail call i32 @_ZN1D1mEi
  ;; --lto-whole-program-visibility disabled but being local this
  ;;  has VCallVisibilityTranslationUnit visibility so it's still devirtualized
  ; CHECK-NO-RTTI-IR: %call3 = tail call i32 @_ZN1D1mEi
  %call3 = tail call i32 %fptr33(ptr nonnull %obj2, i32 %call2)

  ret i32 %call3
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
!1 = !{i64 16, !"_ZTSM1AFviE.virtual"}
!2 = !{i64 24, !"_ZTSM1AFviE.virtual"}
!3 = !{i64 16, !"_ZTS1B"}
!4 = !{i64 16, !"_ZTSM1BFviE.virtual"}
!5 = !{i64 24, !"_ZTSM1BFviE.virtual"}
!6 = !{i64 16, !"_ZTS1C"}
!7 = !{i64 16, !"_ZTSM1CFviE.virtual"}
!8 = !{i64 24, !"_ZTSM1CFviE.virtual"}
!9 = !{i64 16, !10}
!10 = distinct !{}
!11 = !{i64 2}
