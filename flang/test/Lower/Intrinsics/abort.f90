! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPabort_test() {
! CHECK:         %[[VAL_0:.*]] = fir.call @_FortranAAbort() : () -> none
! CHECK:         return
! CHECK:       }

subroutine abort_test()
  call abort
end subroutine
