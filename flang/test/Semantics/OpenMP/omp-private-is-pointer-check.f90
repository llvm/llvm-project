! RUN: %flang_fc1 -fopenmp -fsyntax-only %s

subroutine s
  integer, pointer :: p
  integer, target :: t

  !$omp parallel private(p)
    p=>t
  !$omp end parallel
end subroutine
