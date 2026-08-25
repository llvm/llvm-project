! Valued gang/worker/vector on a loop are only allowed when the loop is
! associated with kernels, and then only when that kernels construct does not
! already specify the matching size clause.

! RUN: split-file %s %t

! Non-kernels: standalone loop
! RUN: not bbc -fopenacc -emit-hlfir %t/loop-gang.f90 -o - 2>&1 | FileCheck %s --check-prefix=LOOP-GANG
! RUN: not bbc -fopenacc -emit-hlfir %t/loop-worker.f90 -o - 2>&1 | FileCheck %s --check-prefix=LOOP-WORKER
! RUN: not bbc -fopenacc -emit-hlfir %t/loop-vector.f90 -o - 2>&1 | FileCheck %s --check-prefix=LOOP-VECTOR

! Non-kernels: combined constructs
! RUN: not bbc -fopenacc -emit-hlfir %t/parallel-loop-gang.f90 -o - 2>&1 | FileCheck %s --check-prefix=PLOOP-GANG
! RUN: not bbc -fopenacc -emit-hlfir %t/parallel-loop-worker.f90 -o - 2>&1 | FileCheck %s --check-prefix=PLOOP-WORKER
! RUN: not bbc -fopenacc -emit-hlfir %t/parallel-loop-vector.f90 -o - 2>&1 | FileCheck %s --check-prefix=PLOOP-VECTOR
! RUN: not bbc -fopenacc -emit-hlfir %t/serial-loop-gang.f90 -o - 2>&1 | FileCheck %s --check-prefix=SLOOP-GANG
! RUN: not bbc -fopenacc -emit-hlfir %t/serial-loop-worker.f90 -o - 2>&1 | FileCheck %s --check-prefix=SLOOP-WORKER
! RUN: not bbc -fopenacc -emit-hlfir %t/serial-loop-vector.f90 -o - 2>&1 | FileCheck %s --check-prefix=SLOOP-VECTOR

! Non-kernels: nested loop inside a compute construct
! RUN: not bbc -fopenacc -emit-hlfir %t/parallel-nested-gang.f90 -o - 2>&1 | FileCheck %s --check-prefix=PAR-GANG
! RUN: not bbc -fopenacc -emit-hlfir %t/parallel-nested-worker.f90 -o - 2>&1 | FileCheck %s --check-prefix=PAR-WORKER
! RUN: not bbc -fopenacc -emit-hlfir %t/parallel-nested-vector.f90 -o - 2>&1 | FileCheck %s --check-prefix=PAR-VECTOR
! RUN: not bbc -fopenacc -emit-hlfir %t/serial-nested-gang.f90 -o - 2>&1 | FileCheck %s --check-prefix=SER-GANG
! RUN: not bbc -fopenacc -emit-hlfir %t/serial-nested-worker.f90 -o - 2>&1 | FileCheck %s --check-prefix=SER-WORKER
! RUN: not bbc -fopenacc -emit-hlfir %t/serial-nested-vector.f90 -o - 2>&1 | FileCheck %s --check-prefix=SER-VECTOR

! Non-kernels: orphan loop in an acc routine
! RUN: not bbc -fopenacc -emit-hlfir %t/routine-gang.f90 -o - 2>&1 | FileCheck %s --check-prefix=RTN-GANG
! RUN: not bbc -fopenacc -emit-hlfir %t/routine-worker.f90 -o - 2>&1 | FileCheck %s --check-prefix=RTN-WORKER
! RUN: not bbc -fopenacc -emit-hlfir %t/routine-vector.f90 -o - 2>&1 | FileCheck %s --check-prefix=RTN-VECTOR

! Kernels with a conflicting size clause
! RUN: not bbc -fopenacc -emit-hlfir %t/kernels-num-gangs.f90 -o - 2>&1 | FileCheck %s --check-prefix=K-NG
! RUN: not bbc -fopenacc -emit-hlfir %t/kernels-num-workers.f90 -o - 2>&1 | FileCheck %s --check-prefix=K-NW
! RUN: not bbc -fopenacc -emit-hlfir %t/kernels-vector-length.f90 -o - 2>&1 | FileCheck %s --check-prefix=K-VL
! RUN: not bbc -fopenacc -emit-hlfir %t/kernels-loop-num-gangs.f90 -o - 2>&1 | FileCheck %s --check-prefix=KL-NG
! RUN: not bbc -fopenacc -emit-hlfir %t/kernels-loop-num-workers.f90 -o - 2>&1 | FileCheck %s --check-prefix=KL-NW
! RUN: not bbc -fopenacc -emit-hlfir %t/kernels-loop-vector-length.f90 -o - 2>&1 | FileCheck %s --check-prefix=KL-VL

! Allowed: kernels without a conflicting size clause, and bare clauses
! RUN: bbc -fopenacc -emit-hlfir %t/allowed.f90 -o - | FileCheck %s --check-prefix=OK

! LOOP-GANG: 'Gang(value)' not allowed in LOOP directive
! LOOP-WORKER: 'Worker(value)' not allowed in LOOP directive
! LOOP-VECTOR: 'Vector(value)' not allowed in LOOP directive
! PLOOP-GANG: 'Gang(value)' not allowed in PARALLEL LOOP directive
! PLOOP-WORKER: 'Worker(value)' not allowed in PARALLEL LOOP directive
! PLOOP-VECTOR: 'Vector(value)' not allowed in PARALLEL LOOP directive
! SLOOP-GANG: 'Gang(value)' not allowed in SERIAL LOOP directive
! SLOOP-WORKER: 'Worker(value)' not allowed in SERIAL LOOP directive
! SLOOP-VECTOR: 'Vector(value)' not allowed in SERIAL LOOP directive
! PAR-GANG: 'Gang(value)' not allowed in PARALLEL LOOP directive
! PAR-WORKER: 'Worker(value)' not allowed in PARALLEL LOOP directive
! PAR-VECTOR: 'Vector(value)' not allowed in PARALLEL LOOP directive
! SER-GANG: 'Gang(value)' not allowed in SERIAL LOOP directive
! SER-WORKER: 'Worker(value)' not allowed in SERIAL LOOP directive
! SER-VECTOR: 'Vector(value)' not allowed in SERIAL LOOP directive
! RTN-GANG: 'Gang(value)' not allowed in subprogram compiled with ROUTINE directive
! RTN-WORKER: 'Worker(value)' not allowed in subprogram compiled with ROUTINE directive
! RTN-VECTOR: 'Vector(value)' not allowed in subprogram compiled with ROUTINE directive
! K-NG: 'Gang(value)' not allowed in KERNELS region that has a NUM_GANGS clause
! K-NW: 'Worker(value)' not allowed in KERNELS region that has a NUM_WORKERS clause
! K-VL: 'Vector(value)' not allowed in KERNELS region that has a VECTOR_LENGTH clause
! KL-NG: 'Gang(value)' not allowed in KERNELS LOOP region that has a NUM_GANGS clause
! KL-NW: 'Worker(value)' not allowed in KERNELS LOOP region that has a NUM_WORKERS clause
! KL-VL: 'Vector(value)' not allowed in KERNELS LOOP region that has a VECTOR_LENGTH clause

!--- loop-gang.f90
subroutine loop_gang(a, n)
  integer :: i, n
  real :: a(n)
  !$acc loop gang(8)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

!--- loop-worker.f90
subroutine loop_worker(a, n)
  integer :: i, n
  real :: a(n)
  !$acc loop worker(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

!--- loop-vector.f90
subroutine loop_vector(a, n)
  integer :: i, n
  real :: a(n)
  !$acc loop vector(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

!--- parallel-loop-gang.f90
subroutine parallel_loop_gang(a, n)
  integer :: i, n
  real :: a(n)
  !$acc parallel loop gang(8)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

!--- parallel-loop-worker.f90
subroutine parallel_loop_worker(a, n)
  integer :: i, n
  real :: a(n)
  !$acc parallel loop worker(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

!--- parallel-loop-vector.f90
subroutine parallel_loop_vector(a, n)
  integer :: i, n
  real :: a(n)
  !$acc parallel loop vector(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

!--- serial-loop-gang.f90
subroutine serial_loop_gang(a, n)
  integer :: i, n
  real :: a(n)
  !$acc serial loop gang(8)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

!--- serial-loop-worker.f90
subroutine serial_loop_worker(a, n)
  integer :: i, n
  real :: a(n)
  !$acc serial loop worker(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

!--- serial-loop-vector.f90
subroutine serial_loop_vector(a, n)
  integer :: i, n
  real :: a(n)
  !$acc serial loop vector(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

!--- parallel-nested-gang.f90
subroutine parallel_nested_gang(a, n)
  integer :: i, n
  real :: a(n)
  !$acc parallel
  !$acc loop gang(8)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end parallel
end subroutine

!--- parallel-nested-worker.f90
subroutine parallel_nested_worker(a, n)
  integer :: i, n
  real :: a(n)
  !$acc parallel
  !$acc loop worker(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end parallel
end subroutine

!--- parallel-nested-vector.f90
subroutine parallel_nested_vector(a, n)
  integer :: i, n
  real :: a(n)
  !$acc parallel
  !$acc loop vector(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end parallel
end subroutine

!--- serial-nested-gang.f90
subroutine serial_nested_gang(a, n)
  integer :: i, n
  real :: a(n)
  !$acc serial
  !$acc loop gang(8)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end serial
end subroutine

!--- serial-nested-worker.f90
subroutine serial_nested_worker(a, n)
  integer :: i, n
  real :: a(n)
  !$acc serial
  !$acc loop worker(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end serial
end subroutine

!--- serial-nested-vector.f90
subroutine serial_nested_vector(a, n)
  integer :: i, n
  real :: a(n)
  !$acc serial
  !$acc loop vector(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end serial
end subroutine

!--- routine-gang.f90
subroutine routine_gang(a, n)
  integer :: i, n
  real :: a(n)
  !$acc routine
  !$acc loop gang(8)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

!--- routine-worker.f90
subroutine routine_worker(a, n)
  integer :: i, n
  real :: a(n)
  !$acc routine
  !$acc loop worker(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

!--- routine-vector.f90
subroutine routine_vector(a, n)
  integer :: i, n
  real :: a(n)
  !$acc routine
  !$acc loop vector(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

!--- kernels-num-gangs.f90
subroutine kernels_num_gangs(a, n)
  integer :: i, n
  real :: a(n)
  !$acc kernels num_gangs(8)
  !$acc loop gang(8)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end kernels
end subroutine

!--- kernels-num-workers.f90
subroutine kernels_num_workers(a, n)
  integer :: i, n
  real :: a(n)
  !$acc kernels num_workers(128)
  !$acc loop worker(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end kernels
end subroutine

!--- kernels-vector-length.f90
subroutine kernels_vector_length(a, n)
  integer :: i, n
  real :: a(n)
  !$acc kernels vector_length(128)
  !$acc loop vector(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end kernels
end subroutine

!--- kernels-loop-num-gangs.f90
subroutine kernels_loop_num_gangs(a, n)
  integer :: i, n
  real :: a(n)
  !$acc kernels loop num_gangs(8) gang(8)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

!--- kernels-loop-num-workers.f90
subroutine kernels_loop_num_workers(a, n)
  integer :: i, n
  real :: a(n)
  !$acc kernels loop num_workers(128) worker(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

!--- kernels-loop-vector-length.f90
subroutine kernels_loop_vector_length(a, n)
  integer :: i, n
  real :: a(n)
  !$acc kernels loop vector_length(128) vector(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

!--- allowed.f90
subroutine allowed_kernels_valued(a, n)
  integer :: i, n
  real :: a(n)
  !$acc kernels
  !$acc loop gang(8)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc loop worker(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc loop vector(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end kernels
end subroutine
! OK-LABEL: func.func @_QPallowed_kernels_valued
! OK: acc.kernels
! OK: acc.loop {{.*}}gang({num=%{{.*}} : i32})
! OK: acc.loop {{.*}}worker(%{{.*}} : i32)
! OK: acc.loop {{.*}}vector(%{{.*}} : i32)

subroutine allowed_kernels_loop_valued(a, n)
  integer :: i, n
  real :: a(n)
  !$acc kernels loop gang(8)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc kernels loop worker(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc kernels loop vector(128)
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine
! OK-LABEL: func.func @_QPallowed_kernels_loop_valued
! OK: acc.kernels {{.*}}combined
! OK: acc.loop {{.*}}gang({num=%{{.*}} : i32})
! OK: acc.loop {{.*}}worker(%{{.*}} : i32)
! OK: acc.loop {{.*}}vector(%{{.*}} : i32)

subroutine allowed_bare_and_size_clause(a, n)
  integer :: i, n
  real :: a(n)
  !$acc parallel loop gang worker vector
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc kernels num_gangs(8) num_workers(128) vector_length(128)
  !$acc loop gang worker vector
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end kernels
end subroutine
! OK-LABEL: func.func @_QPallowed_bare_and_size_clause
! OK: acc.parallel
! OK: acc.loop {{.*}}gang
! OK: acc.kernels {{.*}}num_gangs
! OK: acc.loop {{.*}}gang
