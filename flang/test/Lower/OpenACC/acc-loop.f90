! This test checks lowering of OpenACC loop directive.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s

program acc_loop

  integer :: i, j
  integer, parameter :: n = 10
  real, dimension(n) :: a, b
  real, dimension(n, n) :: c, d


  !$acc loop
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.loop {
!CHECK:        fir.do_loop
!CHECK:        acc.yield
!CHECK-NEXT: }

  !$acc loop collapse(2)
  DO i = 1, n
    DO j = 1, n
      c(i, j) = d(i, j)
    END DO
  END DO

!CHECK:      acc.loop {
!CHECK:        fir.do_loop
!CHECK:          fir.do_loop
!CHECK:        acc.yield
!CHECK-NEXT: } attributes {collapse = 2 : i64}

  !$acc loop
  DO i = 1, n
    !$acc loop
    DO j = 1, n
      c(i, j) = d(i, j)
    END DO
  END DO

!CHECK:      acc.loop {
!CHECK:        fir.do_loop
!CHECK:          acc.loop {
!CHECK:            fir.do_loop
!CHECK:            acc.yield
!CHECK-NEXT:   }
!CHECK:        acc.yield
!CHECK-NEXT: }

end program

