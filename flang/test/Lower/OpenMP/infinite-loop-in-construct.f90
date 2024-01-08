! RUN: bbc -fopenmp -o - %s | FileCheck %s

! Check that this test can be lowered successfully.
! See https://github.com/llvm/llvm-project/issues/74348

! CHECK-LABEL: func.func @_QPsb
! CHECK: omp.parallel
subroutine sb(ninter, numnod)
  integer :: ninter, numnod
  integer, dimension(:), allocatable :: indx_nm

  !$omp parallel
  if (ninter>0) then
    allocate(indx_nm(numnod))
  endif
  220 continue
  goto 220
  !$omp end parallel
end subroutine
