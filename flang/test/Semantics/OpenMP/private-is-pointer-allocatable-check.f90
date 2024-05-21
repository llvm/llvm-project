! UNSUPPORTED: system-windows
! Marking as unsupported due to suspected long runtime on Windows
! RUN: %flang_fc1 -fopenmp -fsyntax-only %s

subroutine s
  integer, pointer :: p
  integer, target :: t
  real(4), allocatable :: arr

  !$omp parallel private(p)
    p=>t
  !$omp end parallel

  allocate(arr)
  !$omp parallel private(arr)
  if (.not. allocated(arr)) then
     print *, 'not allocated'
  endif
  !$omp end parallel
end subroutine
