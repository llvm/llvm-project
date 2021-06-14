! RUN: %S/test_errors.sh %s %t %flang -fopenmp
! REQUIRES: shell
! OpenMP Version 4.5
! 2.7.4 workshare Construct
! Checks for OpenMP Workshare construct

subroutine omp_workshare(aa, bb, cc, dd, ee, ff, n)
  integer i, j, n, a(10), b(10)
  integer, pointer :: p
  integer, target :: t
  real aa(n,n), bb(n,n), cc(n,n), dd(n,n), ee(n,n), ff(n,n)

  !ERROR: The structured block in a WORKSHARE construct may consist of only SCALAR or ARRAY assignments, FORALL or WHERE statements, FORALL, WHERE, ATOMIC, CRITICAL or PARALLEL constructs
  !$omp workshare
  p => t

  !$omp parallel
  cc = dd
  !$omp end parallel

  !ERROR: OpenMP constructs enclosed in WORKSHARE construct may consist of ATOMIC, CRITICAL or PARALLEL constructs only
  !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
  !$omp parallel workshare
  !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
  !$omp single
  ee = ff
  !$omp end single
  !$omp end parallel workshare

  where (aa .ne. 0) cc = bb / aa

  where (b .lt. 2) b = sum(a)

  where (aa .ge. 2.0)
    cc = aa + bb
  elsewhere
    cc = dd + ee
  end where

  forall (i = 1:10, n > i) a(i) = b(i)

  forall (j = 1:10)
    a(j) = a(j) + b(j)
  end forall

  !$omp atomic update
  j = j + sum(a)

  !$omp end workshare

end subroutine omp_workshare
