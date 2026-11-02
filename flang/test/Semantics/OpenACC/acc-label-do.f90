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

subroutine simple_collapse(a, n, m)
  integer :: n, m, jk, jl
  real :: a(n,m)
  !$ACC PARALLEL DEFAULT(PRESENT) ASYNC(1)
  !$ACC LOOP GANG VECTOR COLLAPSE(2)
  DO 214 jk=1,m
    DO 213 jl=1,n
      a(jl,jk) = 1.0
213 END DO
214 END DO
  !$ACC END PARALLEL
end subroutine

subroutine simple_label(c, np)
  integer :: np, n
  real :: c(np)
  !$acc parallel loop present(c)
  do 100 n = 1, np
     c(n) = 0
100 enddo
end subroutine

subroutine shared_end_do(a, n, m)
  integer :: n, m, i, j
  real :: a(n, m)

  !$acc parallel loop collapse(2)
  do 100 j = 1, m
    do 100 i = 1, n
      a(i, j) = 1.0
100 end do
end subroutine

subroutine goto_end_do(c, np)
  integer :: np, n
  real :: c(np)
  !$acc parallel loop
  do 100 n = 1, np
    if (c(n) > 0) goto 100
    c(n) = 0
100 end do
end subroutine

subroutine end_directives(c, np)
  integer :: np, n
  real :: c(np)
  !$acc parallel loop
  do 100 n = 1, np
    c(n) = 0
100 end do
  !$acc end parallel loop
  !$acc parallel
  !$acc loop
  do 200 n = 1, np
    c(n) = 1
200 end do
  !$acc end loop
  !$acc end parallel
end subroutine

subroutine kernels_serial(c, np)
  integer :: np, n
  real :: c(np)
  !$acc kernels loop
  do 100 n = 1, np
    c(n) = 0
100 end do
  !$acc serial loop
  do 200 n = 1, np
    c(n) = 1
200 end do
end subroutine

subroutine named_label_do(c, np)
  integer :: np, n
  real :: c(np)
  !$acc parallel loop
  foo: do 100 n = 1, np
    c(n) = 0
100 end do foo
end subroutine
