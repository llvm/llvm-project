! RUN: bbc %s --pft-test | FileCheck %s
! RUN: bbc %s -o "-" -emit-fir | FileCheck %s --check-prefix=FIR

subroutine sub1()
end subroutine

! CHECK: 1 Subroutine sub1: subroutine sub1()
! CHECK:   1 EndSubroutineStmt: end subroutine
! CHECK: End Subroutine sub1

! FIR-LABEL: func @_QPsub1() {
! FIR:         return
! FIR:       }
