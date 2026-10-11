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

subroutine protected_target
  integer, pointer, protected_target :: p, q
  integer, target :: t

  q => t
  !$omp parallel private(p) firstprivate(q)
    p=>q
  !$omp end parallel
end subroutine
