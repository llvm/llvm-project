! RUN: %python %S/../test_errors.py %s %flang -fopenmp

! A common block in a data-sharing clause is equivalent to listing every
! explicit member of the common block.

subroutine block_then_member()
  common /c/ x, y
  !ERROR: 'x' appears in more than one data-sharing clause on the same OpenMP directive
  !$omp parallel private(/c/) shared(x)
  !$omp end parallel
end subroutine

subroutine member_then_block()
  common /c/ x, y
  !ERROR: 'c' appears in more than one data-sharing clause on the same OpenMP directive
  !$omp parallel shared(x) private(/c/)
  !$omp end parallel
end subroutine

subroutine block_twice()
  common /c/ x, y
  !ERROR: 'c' appears in more than one data-sharing clause on the same OpenMP directive
  !$omp parallel private(/c/) private(/c/)
  !$omp end parallel
end subroutine

subroutine different_blocks()
  common /c/ x
  common /d/ y
  !$omp parallel private(/c/) shared(/d/)
  !$omp end parallel
end subroutine

subroutine separate_directives()
  common /c/ x, y
  !$omp parallel private(/c/)
  !$omp end parallel
  !$omp parallel shared(x)
  !$omp end parallel
end subroutine

! firstprivate and lastprivate may name the same list item.
subroutine fp_lp_block_block()
  common /c/ x, y
  !$omp parallel do firstprivate(/c/) lastprivate(/c/)
  do i = 1, 2
  end do
end subroutine

subroutine fp_member_lp_block()
  common /c/ x, y
  !$omp parallel do firstprivate(x) lastprivate(/c/)
  do i = 1, 2
  end do
end subroutine

subroutine fp_block_lp_member()
  common /c/ x, y
  !$omp parallel do firstprivate(/c/) lastprivate(x)
  do i = 1, 2
  end do
end subroutine

subroutine fp_block_fp_member()
  common /c/ x, y
  !ERROR: 'x' appears in more than one data-sharing clause on the same OpenMP directive
  !$omp parallel do firstprivate(/c/) firstprivate(x)
  do i = 1, 2
  end do
end subroutine
