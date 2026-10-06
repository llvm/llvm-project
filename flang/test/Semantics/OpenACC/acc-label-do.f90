! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! Labeled DO loops that share their terminating label with an inner labeled DO
! loop associated with an OpenACC LOOP construct.

subroutine outer_without_directive(a, n, m)
  integer :: i, k, n, m
  real :: a(n, m)
  do 10 k = 1, m
!$acc loop gang vector
    do 10 i = 1, n
10    a(i, k) = a(i, k) + 1.
end

subroutine outer_without_directive_atomic(a, b, n, m)
  integer :: i, k, n, m
  real :: a(n, m), b(n, m)
  do 20 k = 1, m
!$acc loop gang vector
    do 20 i = 1, n
!$acc atomic update
      a(i, k) = a(i, k) + b(i, k)
!$acc atomic update
20    b(i, k) = b(i, k) + a(i, k)
end

subroutine two_outer_without_directive(a, n, m)
  integer :: i, j, k, n, m
  real :: a(n, m)
  do 30 k = 1, m
    do 30 j = 1, m
!$acc loop
      do 30 i = 1, n
30      a(i, k) = a(i, j)
end

subroutine bad_terminator(n)
  integer :: i, k, n
  do 40 k = 1, n
!$acc loop
    do 40 i = 1, n
!ERROR: This statement cannot terminate the DO loop
!ERROR: This statement cannot terminate the DO loop
40    goto 40
end

subroutine missing_label(a, n)
  integer :: i, k, n
  real :: a(n)
  !ERROR: Label '50' cannot be found
  do 50 k = 1, n
!$acc loop
    do 60 i = 1, n
60    a(i) = 0.
end
