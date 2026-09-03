; Companion to flang/test/Transforms/tbaa-value-dummy-arg.fir.
; Verifies the DSE consequence of the TBAA tag mismatch fixed by that patch.
;
; Flang lowers a VALUE dummy arg as a local alloca copy-store followed by
; loop reads at a runtime offset.  After SROA, DSE asks whether each scalar
; copy-store aliases the loop read.  BasicAA: MayAlias (%off unknown at
; compile time).  DSE consults TBAA:
;
;   Pre-fix:  copy-store "allocated data" (!11) vs loop read
;             "dummy arg data/_QFfnt4Edt" (!9) --> sibling subtrees
;             --> NoAlias --> DSE removes copy-store --> uninitialised read
;
;   Post-fix: copy-store "dummy arg data" (!1) vs same loop read
;             --> ancestor/descendant --> MayAlias --> store kept
;
; Metadata IDs match the post-fix fir-opt --fir-add-alias-tags output for
; flang/test/Transforms/tbaa-value-dummy-arg.fir.
;
; RUN: opt -passes=dse -S %s | FileCheck %s

%T = type { [50 x i8] }

; Pre-fix: copy-store (!11 = "allocated data") vs loop read (!9).
; TBAA NoAlias --> store eliminated.
; CHECK-LABEL: @test_buggy(
; CHECK-NOT:     store i8
; CHECK:         %r = load i8
define i8 @test_buggy(ptr %arg, i64 %off) {
entry:
  %local = alloca %T, align 1
  %v = load i8, ptr %arg,  align 1, !tbaa !9
  store i8 %v, ptr %local, align 1, !tbaa !11
  %p = getelementptr nusw i8, ptr %local, i64 %off
  %r = load i8, ptr %p,    align 1, !tbaa !9
  ret i8 %r
}

; Post-fix: copy-store (!1 = "dummy arg data") vs loop read (!9).
; !2 is ancestor of !10 --> MayAlias --> store kept.
; CHECK-LABEL: @test_fixed(
; CHECK:         store i8 %v
; CHECK:         %r = load i8
define i8 @test_fixed(ptr %arg, i64 %off) {
entry:
  %local = alloca %T, align 1
  %v = load i8, ptr %arg,  align 1, !tbaa !9
  store i8 %v, ptr %local, align 1, !tbaa !1
  %p = getelementptr nusw i8, ptr %local, i64 %off
  %r = load i8, ptr %p,    align 1, !tbaa !9
  ret i8 %r
}

; Flang function root fnt4 (!5)
; └── any access (!4)
;     └── any data access (!3)
;         ├── dummy arg data (!2)
;         │   └── dummy arg data/_QFfnt4Edt (!10)
;         └── allocated data (!8)
!0  = !{i32 2, !"Debug Info Version", i32 3}
!1  = !{!2,  !2,  i64 0}   ; post-fix copy-store tag
!2  = !{!"dummy arg data",            !3, i64 0}
!3  = !{!"any data access",           !4, i64 0}
!4  = !{!"any access",                !5, i64 0}
!5  = !{!"Flang function root fnt4"}
!8  = !{!"allocated data",            !3, i64 0}
!9  = !{!10, !10, i64 0}   ; loop-read tag
!10 = !{!"dummy arg data/_QFfnt4Edt", !2, i64 0}
!11 = !{!8,  !8,  i64 0}   ; pre-fix copy-store tag
