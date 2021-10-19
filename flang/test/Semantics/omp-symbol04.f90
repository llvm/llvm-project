! RUN: %python %S/test_symbols.py %s %flang_fc1 -fopenmp

! 2.15.3 Data-Sharing Attribute Clauses
! Both PARALLEL and DO (worksharing) directives need to create new scope,
! so PRIVATE `a` will have new symbol in each region

  !DEF: /MainProgram1/a ObjectEntity REAL(8)
  real*8 a
  !REF: /MainProgram1/a
  a = 3.14
  !$omp parallel  private(a)
  !DEF: /MainProgram1/Block1/a (OmpPrivate) HostAssoc REAL(8)
  a = 2.
  !$omp do  private(a)
  !DEF: /MainProgram1/Block1/Block1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
     !DEF: /MainProgram1/Block1/Block1/a (OmpPrivate) HostAssoc REAL(8)
     a = 1.
  end do
  !$omp end parallel
  !REF: /MainProgram1/a
  print *, a
end program
