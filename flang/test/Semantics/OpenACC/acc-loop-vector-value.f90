! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! A VECTOR/WORKER clause with a value, or a GANG clause with a num argument,
! is only allowed on a loop associated with KERNELS, and then only when that
! KERNELS construct has no matching size clause (VECTOR_LENGTH/NUM_WORKERS/
! NUM_GANGS). GANG dim and static arguments are allowed on any loop.

subroutine loop_vector(a, n)
  integer :: i, n
  real :: a(n)
  !ERROR: 'Vector(value)' not allowed in LOOP directive
  !$acc loop vector(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

subroutine parallel_loop_vector(a, n)
  integer :: i, n
  real :: a(n)
  !ERROR: 'Vector(value)' not allowed in PARALLEL LOOP directive
  !$acc parallel loop vector(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

subroutine serial_loop_vector(a, n)
  integer :: i, n
  real :: a(n)
  !ERROR: 'Vector(value)' not allowed in SERIAL LOOP directive
  !$acc serial loop vector(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

subroutine parallel_nested_vector(a, n)
  integer :: i, n
  real :: a(n)
  !$acc parallel
  !ERROR: 'Vector(value)' not allowed in PARALLEL LOOP directive
  !$acc loop vector(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end parallel
end subroutine

subroutine serial_nested_vector(a, n)
  integer :: i, n
  real :: a(n)
  !$acc serial
  !ERROR: 'Vector(value)' not allowed in SERIAL LOOP directive
  !$acc loop vector(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end serial
end subroutine

subroutine routine_vector(a, n)
  !$acc routine vector
  integer :: i, n
  real :: a(n)
  !ERROR: 'Vector(value)' not allowed in subprogram compiled with ROUTINE directive
  !$acc loop vector(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

subroutine kernels_vector_length(a, n)
  integer :: i, n
  real :: a(n)
  !$acc kernels vector_length(128)
  !ERROR: 'Vector(value)' not allowed in KERNELS region that has a VECTOR_LENGTH clause
  !$acc loop vector(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end kernels
end subroutine

subroutine kernels_loop_vector_length(a, n)
  integer :: i, n
  real :: a(n)
  !ERROR: 'Vector(value)' not allowed in KERNELS LOOP region that has a VECTOR_LENGTH clause
  !$acc kernels loop vector_length(128) vector(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

subroutine loop_worker(a, n)
  integer :: i, n
  real :: a(n)
  !ERROR: 'Worker(value)' not allowed in LOOP directive
  !$acc loop worker(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

subroutine parallel_loop_worker(a, n)
  integer :: i, n
  real :: a(n)
  !ERROR: 'Worker(value)' not allowed in PARALLEL LOOP directive
  !$acc parallel loop worker(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

subroutine serial_loop_worker(a, n)
  integer :: i, n
  real :: a(n)
  !ERROR: 'Worker(value)' not allowed in SERIAL LOOP directive
  !$acc serial loop worker(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

subroutine parallel_nested_worker(a, n)
  integer :: i, n
  real :: a(n)
  !$acc parallel
  !ERROR: 'Worker(value)' not allowed in PARALLEL LOOP directive
  !$acc loop worker(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end parallel
end subroutine

subroutine serial_nested_worker(a, n)
  integer :: i, n
  real :: a(n)
  !$acc serial
  !ERROR: 'Worker(value)' not allowed in SERIAL LOOP directive
  !$acc loop worker(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end serial
end subroutine

subroutine routine_worker(a, n)
  !$acc routine worker
  integer :: i, n
  real :: a(n)
  !ERROR: 'Worker(value)' not allowed in subprogram compiled with ROUTINE directive
  !$acc loop worker(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

subroutine kernels_num_workers(a, n)
  integer :: i, n
  real :: a(n)
  !$acc kernels num_workers(128)
  !ERROR: 'Worker(value)' not allowed in KERNELS region that has a NUM_WORKERS clause
  !$acc loop worker(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end kernels
end subroutine

subroutine kernels_loop_num_workers(a, n)
  integer :: i, n
  real :: a(n)
  !ERROR: 'Worker(value)' not allowed in KERNELS LOOP region that has a NUM_WORKERS clause
  !$acc kernels loop num_workers(128) worker(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

subroutine loop_gang(a, n)
  integer :: i, n
  real :: a(n)
  !ERROR: 'Gang(value)' not allowed in LOOP directive
  !$acc loop gang(8)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

subroutine parallel_loop_gang(a, n)
  integer :: i, n
  real :: a(n)
  !ERROR: 'Gang(value)' not allowed in PARALLEL LOOP directive
  !$acc parallel loop gang(8)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

subroutine serial_loop_gang(a, n)
  integer :: i, n
  real :: a(n)
  !ERROR: 'Gang(value)' not allowed in SERIAL LOOP directive
  !$acc serial loop gang(num: 8)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

subroutine parallel_nested_gang(a, n)
  integer :: i, n
  real :: a(n)
  !$acc parallel
  !ERROR: 'Gang(value)' not allowed in PARALLEL LOOP directive
  !$acc loop gang(8)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end parallel
end subroutine

subroutine serial_nested_gang(a, n)
  integer :: i, n
  real :: a(n)
  !$acc serial
  !ERROR: 'Gang(value)' not allowed in SERIAL LOOP directive
  !$acc loop gang(8)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end serial
end subroutine

subroutine routine_gang(a, n)
  !$acc routine gang
  integer :: i, n
  real :: a(n)
  !ERROR: 'Gang(value)' not allowed in subprogram compiled with ROUTINE directive
  !$acc loop gang(8)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

subroutine kernels_num_gangs(a, n)
  integer :: i, n
  real :: a(n)
  !$acc kernels num_gangs(8)
  !ERROR: 'Gang(value)' not allowed in KERNELS region that has a NUM_GANGS clause
  !$acc loop gang(8)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end kernels
end subroutine

subroutine kernels_loop_num_gangs(a, n)
  integer :: i, n
  real :: a(n)
  !ERROR: 'Gang(value)' not allowed in KERNELS LOOP region that has a NUM_GANGS clause
  !$acc kernels loop num_gangs(8) gang(8)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

! Size clause after the loop-level clause on the same combined directive.
subroutine kernels_loop_gang_then_num_gangs(a, n)
  integer :: i, n
  real :: a(n)
  !ERROR: 'Gang(value)' not allowed in KERNELS LOOP region that has a NUM_GANGS clause
  !$acc kernels loop gang(8) num_gangs(8)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

subroutine kernels_loop_worker_then_num_workers(a, n)
  integer :: i, n
  real :: a(n)
  !ERROR: 'Worker(value)' not allowed in KERNELS LOOP region that has a NUM_WORKERS clause
  !$acc kernels loop worker(128) num_workers(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

subroutine kernels_loop_vector_then_vector_length(a, n)
  integer :: i, n
  real :: a(n)
  !ERROR: 'Vector(value)' not allowed in KERNELS LOOP region that has a VECTOR_LENGTH clause
  !$acc kernels loop vector(128) vector_length(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

subroutine allowed_kernels_valued(a, n)
  integer :: i, n
  real :: a(n)
  !$acc kernels
  !$acc loop vector(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc loop worker(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc loop gang(8)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc loop vector
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc loop worker
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc loop gang
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end kernels

  !$acc kernels loop vector(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc kernels loop worker(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc kernels loop gang(8)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

subroutine allowed_gang_dim_and_static(a, n)
  integer :: i, n
  real :: a(n)
  !$acc loop gang(dim: 1)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc parallel
  !$acc loop gang(static: *)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end parallel
  !$acc parallel loop gang worker vector
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

! Non-constant integer expressions are still "valued" clauses.
subroutine nonconstant_loop_level_values(a, n, vl, wn, gn)
  integer :: i, n, vl, wn, gn
  real :: a(n)
  !ERROR: 'Vector(value)' not allowed in LOOP directive
  !$acc loop vector(vl)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !ERROR: 'Vector(value)' not allowed in LOOP directive
  !$acc loop vector(length: vl)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !ERROR: 'Worker(value)' not allowed in LOOP directive
  !$acc loop worker(wn)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !ERROR: 'Worker(value)' not allowed in LOOP directive
  !$acc loop worker(num: wn)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !ERROR: 'Gang(value)' not allowed in LOOP directive
  !$acc loop gang(gn)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !ERROR: 'Gang(value)' not allowed in LOOP directive
  !$acc loop gang(num: gn)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !ERROR: 'Vector(value)' not allowed in PARALLEL LOOP directive
  !$acc parallel loop vector(vl + 1)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !ERROR: 'Worker(value)' not allowed in SERIAL LOOP directive
  !$acc serial loop worker(wn * 2)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc parallel
  !ERROR: 'Gang(value)' not allowed in PARALLEL LOOP directive
  !$acc loop gang(num: gn)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end parallel
end subroutine

subroutine nonconstant_routine_vector(a, n, vl)
  !$acc routine vector
  integer :: i, n, vl
  real :: a(n)
  !ERROR: 'Vector(value)' not allowed in subprogram compiled with ROUTINE directive
  !$acc loop vector(vl)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

subroutine nonconstant_kernels_conflicts(a, n, vl, wn, gn)
  integer :: i, n, vl, wn, gn
  real :: a(n)
  !$acc kernels vector_length(vl)
  !ERROR: 'Vector(value)' not allowed in KERNELS region that has a VECTOR_LENGTH clause
  !$acc loop vector(vl)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end kernels

  !$acc kernels num_workers(wn)
  !ERROR: 'Worker(value)' not allowed in KERNELS region that has a NUM_WORKERS clause
  !$acc loop worker(wn)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end kernels

  !$acc kernels num_gangs(gn)
  !ERROR: 'Gang(value)' not allowed in KERNELS region that has a NUM_GANGS clause
  !$acc loop gang(gn)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end kernels

  !ERROR: 'Vector(value)' not allowed in KERNELS LOOP region that has a VECTOR_LENGTH clause
  !$acc kernels loop vector(vl) vector_length(vl)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !ERROR: 'Worker(value)' not allowed in KERNELS LOOP region that has a NUM_WORKERS clause
  !$acc kernels loop worker(num: wn) num_workers(wn)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !ERROR: 'Gang(value)' not allowed in KERNELS LOOP region that has a NUM_GANGS clause
  !$acc kernels loop gang(num: gn) num_gangs(gn)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

subroutine allowed_kernels_nonconstant(a, n, vl, wn, gn)
  integer :: i, n, vl, wn, gn
  real :: a(n)
  !$acc kernels
  !$acc loop vector(vl)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc loop worker(wn)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc loop gang(gn)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc loop vector(length: vl)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc loop worker(num: wn)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc loop gang(num: gn)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end kernels

  !$acc kernels loop vector(vl)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc kernels loop worker(wn)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc kernels loop gang(num: gn)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine
