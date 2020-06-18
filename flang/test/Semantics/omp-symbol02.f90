! RUN: %S/test_symbols.sh %s %t %f18 -fopenmp

! 1.4.1 Structure of the OpenMP Memory Model

! Test implicit declaration in the OpenMP directive enclosing scope
! through clause; also test to avoid creating multiple symbols for
! the same variable

  !DEF: /MainProgram1/b (Implicit) ObjectEntity REAL(4)
  b = 2
  !DEF: /MainProgram1/c (Implicit) ObjectEntity REAL(4)
  c = 0
  !$omp parallel  private(a,b) shared(c,d)
  !DEF: /MainProgram1/Block1/a (OmpPrivate) HostAssoc REAL(4)
  a = 3.
  !DEF: /MainProgram1/Block1/b (OmpPrivate) HostAssoc REAL(4)
  b = 4
  !REF: /MainProgram1/c
  c = 5
  !DEF: /MainProgram1/d (Implicit) ObjectEntity REAL(4)
  d = 6
  !$omp end parallel
  !DEF: /MainProgram1/a (Implicit) ObjectEntity REAL(4)
  print *, a
end program
