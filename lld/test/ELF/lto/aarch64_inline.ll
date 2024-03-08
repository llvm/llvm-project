; REQUIRES: aarch64
;; Test verifies inlining happens cross module when module flags are upgraded
;; by the thin-lto linker/IRMover.
;; Regression test for #82763

; RUN: split-file %s %t
; RUN: opt -thinlto-bc -thinlto-split-lto-unit -unified-lto %t/foo.s -o %t/foo.o
; RUN: opt -thinlto-bc -thinlto-split-lto-unit -unified-lto %t/bar.s -o %t/bar.o
; RUN: opt -thinlto-bc -thinlto-split-lto-unit -unified-lto %t/main.s -o %t/main.o
; RUN: ld.lld -O2 --lto=thin --entry=main %t/main.o %t/foo.o %t/bar.o -o %t/exe
; RUN: llvm-objdump -d %t/exe | FileCheck %s


; CHECK-LABEL:  <main>:
; CHECK-NEXT:         pacibsp
; CHECK-NEXT:         mov     w0, #0x23
; CHECK-NEXT:         autibsp
; CHECK-NEXT:         ret

;--- foo.s
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

define dso_local noundef i32 @foo() local_unnamed_addr #0 {
entry:
  ret i32 34
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) }
!llvm.module.flags = !{!0, !1, !2, !3 }
!0 = !{i32 8, !"branch-target-enforcement", i32 1}
!1 = !{i32 8, !"sign-return-address", i32 1}
!2 = !{i32 8, !"sign-return-address-all", i32 1}
!3 = !{i32 8, !"sign-return-address-with-bkey", i32 1}

;--- bar.s
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

define dso_local noundef i32 @bar() local_unnamed_addr #0 {
entry:
  ret i32 1
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) "branch-target-enforcement"="true" "sign-return-address"="all" "sign-return-address-key"="b_key" }
!llvm.module.flags = !{!0, !1, !2, !3 }
!0 = !{i32 8, !"branch-target-enforcement", i32 2}
!1 = !{i32 8, !"sign-return-address", i32 2}
!2 = !{i32 8, !"sign-return-address-all", i32 2}
!3 = !{i32 8, !"sign-return-address-with-bkey", i32 2}

;--- main.s
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

declare i32 @foo();
declare i32 @bar();

define i32 @main() {
entry:
  %1 = call i32 @foo()
  %2 = call i32 @bar()
  %3 = add i32 %1, %2
  ret i32 %3
}

!llvm.module.flags = !{!0, !1, !2, !3 }
!0 = !{i32 8, !"branch-target-enforcement", i32 1}
!1 = !{i32 8, !"sign-return-address", i32 1}
!2 = !{i32 8, !"sign-return-address-all", i32 1}
!3 = !{i32 8, !"sign-return-address-with-bkey", i32 1}
