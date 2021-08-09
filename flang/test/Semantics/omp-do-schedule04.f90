! RUN: %python %S/test_symbols.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.7.1 Schedule Clause
! Test that does not catch non constant integer expressions like xx - yy.

  !DEF: /tds (Subroutine) Subprogram
subroutine tds
  implicit none
  !DEF: /tds/a ObjectEntity REAL(4)
  !DEF: /tds/y ObjectEntity REAL(4)
  !DEF: /tds/z ObjectEntity REAL(4)
  real a(100),y(100),z(100)
  !DEF: /tds/i ObjectEntity INTEGER(4)
  !DEF: /tds/j ObjectEntity INTEGER(4)
  !DEF: /tds/k ObjectEntity INTEGER(4)
  integer i,j,k

  !REF: /tds/j
  j = 11
  !REF: /tds/k
  k = 12
  !$omp do  schedule(static,j-k)
  !DEF: /tds/Block1/i (OmpPrivate,OmpPreDetermined) HostAssoc INTEGER(4)
  do i = 1,10
    !REF: /tds/y
    !REF: /tds/Block1/i
    !REF: /tds/z
    !REF: /tds/a
    y(i) = z(i-1)+a(i)
  end do
  !$omp end do
end subroutine tds
