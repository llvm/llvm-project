! RUN: %flang_fc1 -fopenmp -fsyntax-only %s

! Check that using 'present' inside 'parallel' doesn't cause syntax errors.
subroutine test_present(opt)
  integer, optional :: opt
  !$omp parallel
    if (present(opt)) print *, "present"
  !$omp end parallel
end subroutine
