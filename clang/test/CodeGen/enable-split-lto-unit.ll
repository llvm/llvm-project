; Test that we do not duplicate the EnableSplitLTOUnit module flag.
;
; Disable the verifier so the compiler doesn't abort and thus lead to empty
; output and false pass.
;
; RUN: %clang_cc1 -fno-legacy-pass-manager -emit-llvm-bc -flto=full -disable-llvm-verifier -o - %s | llvm-dis | FileCheck %s --check-prefix=FULL-NPM
; RUN: %clang_cc1 -fno-legacy-pass-manager -emit-llvm-bc -flto=thin -disable-llvm-verifier -o - %s | llvm-dis | FileCheck %s --check-prefix=THIN-NPM
; RUN: %clang_cc1 -flegacy-pass-manager -emit-llvm-bc -flto=full -disable-llvm-verifier -o - %s | llvm-dis | FileCheck %s --check-prefix=FULL-OPM
; RUN: %clang_cc1 -flegacy-pass-manager -emit-llvm-bc -flto=thin -disable-llvm-verifier -o - %s | llvm-dis | FileCheck %s --check-prefix=THIN-OPM

define dso_local void @main() local_unnamed_addr {
entry:
  ret void
}

; FULL-NPM-NOT: !llvm.module.flags = !{!0, !1, !2, !3, !3}
; FULL-OPM-NOT: !llvm.module.flags = !{!0, !1, !2, !3, !3}
; THIN-NPM-NOT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
; THIN-OPM-NOT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"wchar_size", i32 2}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, !"ThinLTO", i32 0}
!3 = !{i32 1, !"EnableSplitLTOUnit", i32 1}
