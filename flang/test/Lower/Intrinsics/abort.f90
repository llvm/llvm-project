! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPabort_test() {
! CHECK:         fir.call @_FortranAAbort() {{.*}}: () -> ()
! CHECK-NEXT:    fir.unreachable


subroutine abort_test()
  call abort
end subroutine
