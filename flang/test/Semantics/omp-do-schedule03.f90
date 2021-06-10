! RUN: %S/test_symbols.sh %s %t %flang_fc1 -fopenmp
! REQUIRES: shell
! OpenMP Version 4.5
! 2.7.1 Schedule Clause
! Test that does not catch non constant integer expressions like xx - xx.
  !DEF: /ompdoschedule MainProgram
program ompdoschedule
  !DEF: /ompdoschedule/a ObjectEntity REAL(4)
  !DEF: /ompdoschedule/y ObjectEntity REAL(4)
  !DEF: /ompdoschedule/z ObjectEntity REAL(4)
  real  a(100),y(100),z(100)
  !DEF: /ompdoschedule/b ObjectEntity INTEGER(4)
  !DEF: /ompdoschedule/i ObjectEntity INTEGER(4)
  !DEF: /ompdoschedule/n ObjectEntity INTEGER(4)
  integer  b,i,n
  !REF: /ompdoschedule/b
  b = 10
  !$omp do  schedule(static,b-b)
  !DEF: /ompdoschedule/Block1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  !REF: /ompdoschedule/n
  do i = 2,n+1
    !REF: /ompdoschedule/y
    !REF: /ompdoschedule/Block1/i
    !REF: /ompdoschedule/z
    !REF: /ompdoschedule/a
    y(i) = z(i-1) + a(i)
  end do
  !$omp end do
end program ompdoschedule
