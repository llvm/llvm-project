! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! Fortran 2008 and 2018 require that a compiler implement localization of
! variables when the iterations of a DO CONCURRENT loop are executed in
! parallel.
! Specifically, F'2018 (11.1.7.5 para 4) "if a variable has unspecified
! locality, (first bullet) if it is referenced in an iteration it shall
! either be previously defined during that iteration, or shall not be
! defined or become undefined during any other iteration; ...".
! This clause implies that it is well defined for a variable to be defined
! and then referenced in the same iteration, even if it is also defined by
! other iterations.  Consequently, if a compiler executes the iterations of
! the DO CONCURRENT loop in parallel, it must effectively localize some
! variables that do not (or cannot) appear in a locality clause.
! This is not a problem that can be wholly resolved at compilation time in
! all cases, as is demonstrated below.

program main
  implicit none
  integer, parameter :: N=1000
  integer :: J, K(N), L(N)
  real :: A(N), B(N), T(N)
  do J=1,N
    K(J) = 1
    L(J) = 1
    A(J) = J
  end do
  call foo(N, A, B, T, K, L)
  do J=1,N
    if (B(J) /= A(J)) then
      print *, 'FAIL at ', J, ' expected ', A(J), ' got ', B(J)
      stop
    end if
  end do
  print *, 'PASS'
end program main

subroutine foo(N, A, B, T, K, L)
  implicit none
  integer, intent(in) :: N, K(N), L(N)
  real, intent(in) :: A(N)
  real, intent(out) :: B(N)
  real, intent(inout) :: T(N)
  integer :: J
  do concurrent (J=1:N)
    ! During execution, K(J) and L(J) are both always 1.  So the store
    ! and load to/from T() always affect T(1) in each iteration.  Since
    ! T(1) is defined in each iteration before it is referenced, this
    ! program conforms with F2008 and F2018.
    T(K(J)) = A(J)
    B(J) = T(L(J)) ! must be A(J) whenever K(J)==L(J)
  end do
end subroutine foo
