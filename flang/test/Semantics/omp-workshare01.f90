! RUN: %python %S/test_errors.py %s %flang -fopenmp
! OpenMP Version 4.5
! 2.7.4 workshare Construct
! Invalid do construct inside !$omp workshare

subroutine workshare(aa, bb, cc, dd, ee, ff, n)
  integer n, i
  real aa(n,n), bb(n,n), cc(n,n), dd(n,n), ee(n,n), ff(n,n)

  !ERROR: The structured block in a WORKSHARE construct may consist of only SCALAR or ARRAY assignments, FORALL or WHERE statements, FORALL, WHERE, ATOMIC, CRITICAL or PARALLEL constructs
  !ERROR: OpenMP constructs enclosed in WORKSHARE construct may consist of ATOMIC, CRITICAL or PARALLEL constructs only
  !$omp workshare
  do i = 1, n
    print *, "omp workshare"
  end do

  !$omp critical
  !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
  !$omp single
  aa = bb
  !$omp end single
  !$omp end critical

  !$omp parallel
  !$omp single
  cc = dd
  !$omp end single
  !$omp end parallel

  ee = ff
  !$omp end workshare

end subroutine workshare
