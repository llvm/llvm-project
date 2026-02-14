! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

integer function test_exit(status)
  integer, intent(in) :: status
  if (status > 0) call exit(status)
  if (status == 42) print *, "Unreachable"
  test_exit = status
end function test_exit

! CHECK-LABEL: func.func @_QPtest_exit
! CHECK: fir.call @_FortranAExit
! CHECK-NEXT: fir.unreachable
! CHECK: func.func private @_FortranAExit{{.*}} attributes {{.*}}noreturn
