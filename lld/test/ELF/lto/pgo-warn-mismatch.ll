; REQUIRES: x86

; RUN: rm -rf %t && split-file %s %t
; RUN: opt -module-summary %t/a.ll -o %t/a.bc
; RUN: llvm-profdata merge %t/cs.proftext -o %t/cs.profdata

;; Ensure lld generates warnings for profile cfg mismatch.
; RUN: ld.lld --lto-cs-profile-file=%t/cs.profdata --lto-pgo-warn-mismatch -shared %t/a.bc -o /dev/null 2>&1 | FileCheck %s

;; Ensure lld will not generate warnings for profile cfg mismatch.
; RUN: ld.lld --lto-cs-profile-file=%t/cs.profdata --no-lto-pgo-warn-mismatch -shared --fatal-warnings %t/a.bc -o /dev/null

; CHECK: warning: {{.*}} function control flow change detected (hash mismatch) f Hash = [[#]]

;--- a.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @f(i32 returned %a) #0 {
entry:
  ret i32 %a
}

attributes #0 = { "target-cpu"="x86-64" }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10, !11, !12}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 2}
!5 = !{!"MaxCount", i64 1}
!6 = !{!"MaxInternalCount", i64 0}
!7 = !{!"MaxFunctionCount", i64 1}
!8 = !{!"NumCounts", i64 3}
!9 = !{!"NumFunctions", i64 2}
!10 = !{!"IsPartialProfile", i64 0}
!11 = !{!"PartialProfileRatio", double 0.000000e+00}
!12 = !{!"DetailedSummary", !13}
!13 = !{!14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29}
!14 = !{i32 10000, i64 0, i32 0}
!15 = !{i32 100000, i64 0, i32 0}
!16 = !{i32 200000, i64 0, i32 0}
!17 = !{i32 300000, i64 0, i32 0}
!18 = !{i32 400000, i64 0, i32 0}
!19 = !{i32 500000, i64 1, i32 2}
!20 = !{i32 600000, i64 1, i32 2}
!21 = !{i32 700000, i64 1, i32 2}
!22 = !{i32 800000, i64 1, i32 2}
!23 = !{i32 900000, i64 1, i32 2}
!24 = !{i32 950000, i64 1, i32 2}
!25 = !{i32 990000, i64 1, i32 2}
!26 = !{i32 999000, i64 1, i32 2}
!27 = !{i32 999900, i64 1, i32 2}
!28 = !{i32 999990, i64 1, i32 2}
!29 = !{i32 999999, i64 1, i32 2}

;--- cs.proftext
# CSIR level Instrumentation Flag
:csir
f
# Func Hash:
1535914979662757887
# Num Counters:
2
# Counter Values:
1
0

