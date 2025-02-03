! RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenmp

! OpenMP 5.2 5.1.1 Variables Referenced in a Construct
! FORALL indices have predetermined private DSA.
!
! As FORALL indices are defined in the construct itself, and OpenMP
! directives may not appear in it, they are already private.
! Check that index symbols are not modified.

  !DEF: /MainProgram1/a ObjectEntity INTEGER(4)
  !DEF: /MainProgram1/b ObjectEntity INTEGER(4)
  integer a(5), b(5)

  !REF: /MainProgram1/a
  a = 0
  !REF: /MainProgram1/b
  b = 0

  !$omp parallel
    !DEF: /MainProgram1/OtherConstruct1/Forall1/i (Implicit) ObjectEntity INTEGER(4)
    !DEF: /MainProgram1/OtherConstruct1/a HostAssoc INTEGER(4)
    !DEF: /MainProgram1/OtherConstruct1/b HostAssoc INTEGER(4)
    forall(i = 1:5) a(i) = b(i) * 2
  !$omp end parallel

  !$omp parallel default(private)
    !DEF: /MainProgram1/OtherConstruct2/Forall1/i (Implicit) ObjectEntity INTEGER(4)
    !DEF: /MainProgram1/OtherConstruct2/a (OmpPrivate) HostAssoc INTEGER(4)
    !DEF: /MainProgram1/OtherConstruct2/b (OmpPrivate) HostAssoc INTEGER(4)
    forall(i = 1:5) a(i) = b(i) * 2
  !$omp end parallel
end program
