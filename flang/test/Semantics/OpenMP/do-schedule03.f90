! RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.7.1 Schedule Clause
! Test that does not catch non constant integer expressions like xx - xx.
  !DEF: /OMPDOSCHEDULE MainProgram
program OMPDOSCHEDULE
  !DEF: /OMPDOSCHEDULE/a ObjectEntity REAL(4)
  !DEF: /OMPDOSCHEDULE/y ObjectEntity REAL(4)
  !DEF: /OMPDOSCHEDULE/z ObjectEntity REAL(4)
  real  a(100),y(100),z(100)
  !DEF: /OMPDOSCHEDULE/b ObjectEntity INTEGER(4)
  !DEF: /OMPDOSCHEDULE/i ObjectEntity INTEGER(4)
  !DEF: /OMPDOSCHEDULE/n ObjectEntity INTEGER(4)
  integer  b,i,n
  !REF: /OMPDOSCHEDULE/b
  b = 10
  !$omp do  schedule(static,b-b)
  !DEF: /OMPDOSCHEDULE/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  !REF: /OMPDOSCHEDULE/n
  do i = 2,n+1
    !REF: /OMPDOSCHEDULE/y
    !REF: /OMPDOSCHEDULE/OtherConstruct1/i
    !REF: /OMPDOSCHEDULE/z
    !REF: /OMPDOSCHEDULE/a
    y(i) = z(i-1) + a(i)
  end do
  !$omp end do
end program OMPDOSCHEDULE
