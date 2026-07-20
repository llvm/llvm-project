! Test that the VectorAlwaysUnroll pass is scheduled in the FIR optimizer
! pipeline at -O2 by default, and is skipped when LLVM's VPlan-native
! outer-loop vectorization path is enabled (-enable-vplan-native-path).
!
! The pass tags inner loops for full unrolling so the regular loop vectorizer
! can vectorize the annotated outer loop. When the VPlan-native path is
! available it can vectorize outer loops directly, so this workaround is
! unnecessary and must be skipped.

! RUN: %flang_fc1 -S -O2 -mmlir --mlir-pass-statistics -mmlir --mlir-pass-statistics-display=pipeline -o /dev/null %s 2>&1 | FileCheck --check-prefix=DEFAULT %s
! RUN: %flang_fc1 -S -O2 -mllvm -enable-vplan-native-path -mmlir --mlir-pass-statistics -mmlir --mlir-pass-statistics-display=pipeline -o /dev/null %s 2>&1 | FileCheck --check-prefix=VPLAN %s

! REQUIRES: asserts

end program

! Default (no -enable-vplan-native-path): the pass is scheduled.
! DEFAULT: Pass statistics report
! DEFAULT: VectorAlwaysUnroll

! With -enable-vplan-native-path: the pass is skipped.
! VPLAN: Pass statistics report
! VPLAN-NOT: VectorAlwaysUnroll
