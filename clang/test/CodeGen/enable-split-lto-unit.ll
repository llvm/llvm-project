; Test that we do not duplicate the EnableSplitLTOUnit module flag.
;
; Disable the verifier so the compiler doesn't abort and thus lead to empty
; output and false pass.
;
; RUN: %clang_cc1 -emit-llvm-bc -flto=full -disable-llvm-verifier -o - %s | llvm-dis | FileCheck %s --check-prefix=FULL
; RUN: %clang_cc1 -emit-llvm-bc -flto=thin -disable-llvm-verifier -o - %s | llvm-dis | FileCheck %s --check-prefix=THIN

define dso_local void @main() local_unnamed_addr {
entry:
  ret void
}

; FULL-NOT: !llvm.module.flags = !{!0, !1, !2, !3, !3}
; THIN-NOT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"wchar_size", i32 2}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, !"ThinLTO", i32 0}
!3 = !{i32 1, !"EnableSplitLTOUnit", i32 1}
