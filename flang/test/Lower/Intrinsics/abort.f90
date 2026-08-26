! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPabort_test() {
! CHECK:         fir.call @_FortranAAbort() {{.*}}: () -> ()
! CHECK-NEXT:    fir.unreachable
! CHECK-NEXT:    }
! CHECK: func.func private @_FortranAAbort{{.*}} attributes {{.*}}noreturn

subroutine abort_test()
  call abort
end subroutine
