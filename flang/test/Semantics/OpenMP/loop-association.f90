! RUN: %python %S/../test_errors.py %s %flang -fopenmp

! Check the association between OpenMPLoopConstruct and DoConstruct

  integer :: b = 128
  integer :: c = 32
  integer, parameter :: num = 16
  N = 1024

! Different DO loops

  !$omp parallel
  !$omp do
  do 10 i=1, N
     a = 3.14
10   print *, a
  !$omp end parallel

  !$omp parallel do
  !ERROR: DO CONCURRENT loops cannot form part of a loop nest.
  DO CONCURRENT (i = 1:N)
     a = 3.14
  END DO

  !$omp parallel do simd
  !ERROR: The associated loop of a loop-associated directive cannot be a DO WHILE.
  outer: DO WHILE (c > 1)
     inner: do while (b > 100)
        a = 3.14
        b = b - 1
     enddo inner
     c = c - 1
  END DO outer

  ! Accept directives between parallel do and actual loop.
  !$OMP PARALLEL DO
  !WARNING: Unrecognized compiler directive was ignored [-Wignored-directive]
  !WARNING: Compiler directives are not allowed inside OpenMP loop constructs
  !DIR$ VECTOR ALIGNED
  DO 20 i=1,N
     a = a + 0.5
20   CONTINUE
  !$OMP END PARALLEL DO

  c = 16
  !$omp parallel do
  !ERROR: Loop control is not present in the DO LOOP
  !ERROR: The associated loop of a loop-associated directive cannot be a DO without control.
  do
     a = 3.14
     c = c - 1
     !ERROR: EXIT to construct outside of PARALLEL DO construct is not allowed
     !ERROR: EXIT statement terminates associated loop of an OpenMP DO construct
     if (c < 1) exit
  enddo

! Loop association check

  ! If an end do directive follows a do-construct in which several DO
  ! statements share a DO termination statement, then a do directive
  ! can only be specified for the outermost of these DO statements.
  do 100 i=1, N
     !$omp do
     do 100 j=1, N
        a = 3.14
100     continue
    !ERROR: END DO directive is not allowed when the construct does not contain all loops that share a loop-terminating statement
    !$omp enddo

  !ERROR: Non-THREADPRIVATE object 'a' in COPYIN clause
  !$omp parallel do copyin(a)
  do i = 1, N
     !$omp parallel do
     do j = 1, i
     enddo
     !$omp end parallel do
     a = 3.
  enddo
  !$omp end parallel do

  !$omp parallel do
  do i = 1, N
  enddo
  !$omp end parallel do
  !ERROR: Misplaced OpenMP end-directive
  !$omp end parallel do

  !$omp parallel
  a = 3.0
  !$omp do simd
  do i = 1, N
  enddo
  !$omp end do simd

  !ERROR: Non-THREADPRIVATE object 'a' in COPYIN clause
  !$omp parallel do copyin(a)
  do i = 1, N
  enddo
  !$omp end parallel

  a = 0.0
  !ERROR: Misplaced OpenMP end-directive
  !$omp end parallel do
  !$omp parallel do private(c)
  do i = 1, N
     do j = 1, N
        !ERROR: OpenMP loop construct should contain a DO-loop or a loop-nest-generating OpenMP construct
        !$omp parallel do shared(b)
        a = 3.14
     enddo
     !ERROR: Misplaced OpenMP end-directive
     !$omp end parallel do
  enddo
  a = 1.414
  !ERROR: Misplaced OpenMP end-directive
  !$omp end parallel do

  do i = 1, N
     !$omp parallel do
     do j = 2*i*N, (2*i+1)*N
        a = 3.14
     enddo
  enddo
  !ERROR: Misplaced OpenMP end-directive
  !$omp end parallel do

  !ERROR: OpenMP loop construct should contain a DO-loop or a loop-nest-generating OpenMP construct
  !$omp parallel do private(c)
5 FORMAT (1PE12.4, I10)
  do i=1, N
     a = 3.14
  enddo
  !ERROR: Misplaced OpenMP end-directive
  !$omp end parallel do

  !$omp parallel do simd
  do i = 1, N
     a = 3.14
  enddo
  !$omp end parallel do simd
  !ERROR: Misplaced OpenMP end-directive
  !$omp end parallel do simd

  !ERROR: OpenMP loop construct should contain a DO-loop or a loop-nest-generating OpenMP construct
  !$omp simd
    a = i + 1
  !ERROR: Misplaced OpenMP end-directive
  !$omp end simd
   a = i + 1
end
