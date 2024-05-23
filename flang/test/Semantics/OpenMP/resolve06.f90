! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags
use omp_lib
!2.11.4 Allocate Clause
!For any list item that is specified in the allocate
!clause on a directive, a data-sharing attribute clause
!that may create a private copy of that list item must be
!specified on the same directive.

  integer ::  N = 2

  !ERROR: The ALLOCATE clause requires that 'x' must be listed in a private data-sharing attribute clause on the same directive
  !$omp parallel allocate(omp_default_mem_space : x)
  do i = 1, N
     x = 2
  enddo
  !$omp end parallel

  !ERROR: The ALLOCATE clause requires that 'y' must be listed in a private data-sharing attribute clause on the same directive
  !$omp parallel allocate(omp_default_mem_space : y) firstprivate(x)
  do i = 1, N
     x = 2
  enddo
  !$omp end parallel

  !ERROR: The ALLOCATE clause requires that 'x' must be listed in a private data-sharing attribute clause on the same directive
  !ERROR: The ALLOCATE clause requires that 'x' must be listed in a private data-sharing attribute clause on the same directive
  !$omp parallel allocate(omp_default_mem_space : x) allocate(omp_default_mem_space : x)
  do i = 1, N
     x = 2
  enddo
  !$omp end parallel

  !ERROR: The ALLOCATE clause requires that 'f' must be listed in a private data-sharing attribute clause on the same directive
  !$omp parallel allocate(omp_default_mem_space : f) shared(f)
  do i = 1, N
     x = 2
  enddo
  !$omp end parallel

  !ERROR: The ALLOCATE clause requires that 'q' must be listed in a private data-sharing attribute clause on the same directive
  !$omp parallel private(t) allocate(omp_default_mem_space : z, t, q, r) firstprivate(z, r)
  do i = 1, N
     x = 2
  enddo
  !$omp end parallel

  !ERROR: The ALLOCATE clause requires that 'b' must be listed in a private data-sharing attribute clause on the same directive
  !ERROR: The ALLOCATE clause requires that 'c' must be listed in a private data-sharing attribute clause on the same directive
  !$omp parallel allocate(omp_default_mem_space : a, b, c, d) firstprivate(a, d)
  do i = 1, N
     x = 2
  enddo
  !$omp end parallel
end
