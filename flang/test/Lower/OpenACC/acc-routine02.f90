! This test checks lowering of OpenACC routine directive.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine sub1(a, n)
  integer :: n
  real :: a(n)
end subroutine sub1

!$acc routine(sub1)

program test
  integer, parameter :: N = 10
  real :: a(N)
  call sub1(a, N)
end program

! CHECK-LABEL: acc.routine @acc_routine_0 func(@_QPsub1)

! CHECK: func.func @_QPsub1(%ar{{.*}}: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_0]>} 
