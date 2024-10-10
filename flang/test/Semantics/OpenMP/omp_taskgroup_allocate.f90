! RUN: %flang_fc1 -fopenmp -fopenmp-version=50 -fsyntax-only %s

! Verify that a list item with a private data-sharing clause used in the 'allocate' clause of 'taskgroup'
! causes no semantic errors.

subroutine omp_allocate_taskgroup
   integer :: x
   !$omp parallel private(x)
   !$omp taskgroup allocate(x)
   !$omp end taskgroup
   !$omp end parallel
end subroutine
