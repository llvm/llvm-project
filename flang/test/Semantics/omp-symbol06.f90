! RUN: %python %S/test_symbols.py %s %flang_fc1 -fopenmp

! 2.15.3 Data-Sharing Attribute Clauses
! A list item that specifies a given variable may not appear in more than
! one clause on the same directive, except that a variable may be specified
! in both firstprivate and lastprivate clauses.

  !DEF: /MainProgram1/a (Implicit) ObjectEntity REAL(4)
  a = 1.
  !$omp parallel do  firstprivate(a) lastprivate(a)
  !DEF: /MainProgram1/Block1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
     !DEF: /MainProgram1/Block1/a (OmpFirstPrivate, OmpLastPrivate) HostAssoc REAL(4)
     a = 2.
  end do
end program
