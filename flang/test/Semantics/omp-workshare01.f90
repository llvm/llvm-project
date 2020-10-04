! RUN: %S/test_errors.sh %s %t %f18 -fopenmp
! XFAIL: *

! OpenMP Version 4.5
! 2.7.4 workshare Construct
! Invalid do construct inside !$omp workshare

subroutine workshare(aa, bb, cc, dd, ee, ff, n)
  integer n, i
  real aa(n,n), bb(n,n), cc(n,n), dd(n,n), ee(n,n), ff(n,n)

  !$omp workshare
  !ERROR: Unexpected do stmt inside !$omp workshare
  do i = 1, n
    print *, "omp workshare"
  end do

  aa = bb
  cc = dd
  ee = ff
  !$omp end workshare

end subroutine workshare
