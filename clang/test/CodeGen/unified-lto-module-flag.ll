; Test that we do not duplicate the UnifiedLTO module flag.
;
; RUN: %clang_cc1 -emit-llvm -flto=full -funified-lto -o - %s | FileCheck %s

; CHECK: !llvm.module.flags = !{!0, !1, !2, !3}
!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"wchar_size", i32 2}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, !"EnableSplitLTOUnit", i32 1}
!3 = !{i32 1, !"UnifiedLTO", i32 1}
