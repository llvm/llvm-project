! RUN: %S/test_symbols.sh %s %t %flang_fc1 -fopenmp
! REQUIRES: shell

! 1.4.1 Structure of the OpenMP Memory Model
! In the inner OpenMP region, SHARED `a` refers to the `a` in the outer OpenMP
! region; PRIVATE `b` refers to the new `b` in the same OpenMP region

  !DEF: /MainProgram1/b (Implicit) ObjectEntity REAL(4)
  b = 2
  !$omp parallel  private(a) shared(b)
  !DEF: /MainProgram1/Block1/a (OmpPrivate) HostAssoc REAL(4)
  a = 3.
  !REF: /MainProgram1/b
  b = 4
  !$omp parallel  private(b) shared(a)
  !REF: /MainProgram1/Block1/a
  a = 5.
  !DEF: /MainProgram1/Block1/Block1/b (OmpPrivate) HostAssoc REAL(4)
  b = 6
  !$omp end parallel
  !$omp end parallel
  !DEF: /MainProgram1/a (Implicit) ObjectEntity REAL(4)
  !REF: /MainProgram1/b
  print *, a, b
end program
