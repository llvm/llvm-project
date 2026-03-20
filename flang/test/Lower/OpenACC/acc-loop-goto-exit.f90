! Test that GoTo exiting an acc.loop seq region generates acc.yield
! instead of an illegal cross-region cf.br that would crash the compiler.
! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPfoo
subroutine foo(N, A, B)
  implicit real*8 (a-h, o-z)
  !$acc routine gang
  dimension A(*), B(*)
  !$acc loop gang vector
  do 100 i = 1, N
  ! CHECK: acc.loop gang vector
  ! CHECK: acc.loop {{.*}} {
  !$acc loop seq
    do 10 j = 1, 1000
      if (A(i) .gt. B(i)) goto 20
10  continue
  ! The GoTo crossing the acc.loop region boundary must generate
  ! acc.yield to properly exit the inner acc.loop, not an illegal
  ! cross-region cf.br that would crash the compiler.
  ! CHECK: acc.yield
  ! CHECK: acc.yield
  ! CHECK: }
20  B(i) = A(i)
100 continue
end subroutine
