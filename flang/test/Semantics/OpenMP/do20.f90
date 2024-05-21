! RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenmp

! OpenMP 5.2 5.1.1
! Iteration variables of non-associated loops may be listed in DSA clauses.

!DEF: /shared_iv (Subroutine)Subprogram
subroutine shared_iv
  !DEF: /shared_iv/i ObjectEntity INTEGER(4)
  integer i

  !$omp parallel shared(i)
    !$omp single
      !REF: /shared_iv/i
      do i = 0, 1
      end do
    !$omp end single
  !$omp end parallel
end subroutine
