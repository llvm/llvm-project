! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine test_unstructured1(a, b, c)
  integer :: i, j, k
  real :: a(:,:,:), b(:,:,:), c(:,:,:)

  !$acc data copy(a, b, c)

  !$acc kernels
  a(:,:,:) = 0.0
  !$acc end kernels

  !$acc kernels
  do i = 1, 10
    do j = 1, 10
      do k = 1, 10
      end do
    end do
  end do
  !$acc end kernels

  do i = 1, 10
    do j = 1, 10
      do k = 1, 10
      end do
    end do

    if (a(1,2,3) > 10) stop 'just to be unstructured'
  end do

  !$acc end data

end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured1
! CHECK: acc.data
! CHECK: acc.kernels
! CHECK: acc.kernels
! CHECK: fir.call @_FortranAStopStatementText


! Tests with independent loops + unstructured CFG (which currently hit the
! NYI emitted in genACC for `acc loop` / combined constructs) live as
! TODO-style tests under flang/test/Lower/OpenACC/Todo/:
!   - Todo/acc-unstructured-combined-construct.f90  (`acc parallel loop` ...)
!   - Todo/acc-unstructured-loop-construct.f90      (standalone `acc loop` ...)

subroutine test_unstructured3(a, b, c)
  integer :: i, j, k
  real :: a(:,:,:), b(:,:,:), c(:,:,:)

  !$acc parallel
  do i = 1, 10
    do j = 1, 10
      do k = 1, 10
        if (a(1,2,3) > 10) stop 'just to be unstructured'
      end do
    end do
  end do
  !$acc end parallel

! CHECK-LABEL: func.func @_QPtest_unstructured3
! CHECK: acc.parallel
! CHECK: fir.call @_FortranAStopStatementText
! CHECK: acc.yield
! CHECK: acc.yield

end subroutine

! Test that acc.data is still created when there are no data clauses but the
! construct contains unstructured control flow. Without this, the early return
! in genACCDataOp skips acc.data creation, leaving orphaned blocks.
subroutine test_unstructured4(a, n)
  integer :: n, i, j
  real :: a(:)
  logical :: use_gpu

  use_gpu = .true.
  !$acc data if(use_gpu)
  do i = 1, n
    do j = 1, n
      if (a(j) > 0.0) stop 'unstructured'
    end do
  end do
  !$acc end data

end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured4
! CHECK: acc.data if(%{{.*}}) {
! CHECK: fir.call @_FortranAStopStatementText
! CHECK: acc.terminator
! CHECK: }

! Test that GOTO exiting acc.data (one level) generates acc.terminator
! instead of an invalid cross-region branch.
subroutine test_unstructured5(a, n)
  integer :: n, i, j
  real :: a(:)
  logical :: use_gpu

  use_gpu = .true.
  !$acc data if(use_gpu)
  do i = 1, n
    do j = 1, n
      if (a(j) > 0.0) goto 999
    end do
  end do
  !$acc end data
999 continue

end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured5
! CHECK: acc.data if(%{{.*}}) {
! CHECK: fir.store %{{.*}} to %{{.*}} : !fir.ref<i32>
! CHECK: acc.terminator
! CHECK: acc.terminator
! CHECK: }
! CHECK: arith.cmpi eq
! CHECK: cf.cond_br

! Test GOTO exiting acc.data with intermediate code. Jump table dispatches
! after the acc.data op.
subroutine test_unstructured8(a, n)
  integer :: n, i, j
  real :: a(:)
  logical :: use_gpu
  use_gpu = .true.
  !$acc data if(use_gpu)
  do i = 1, n
    do j = 1, n
      if (a(j) > 0.0) goto 999
    end do
  end do
  a(1) = -1.0
  !$acc end data
999 continue
end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured8
! Inside acc.data, GOTO stores exit selector and terminates:
! CHECK: acc.data if(%{{.*}}) {
! CHECK: fir.store %{{.*}} to %{{.*}} : !fir.ref<i32>
! CHECK: acc.terminator
! CHECK: acc.terminator
! CHECK: }
! Jump table after acc.data:
! CHECK: fir.load %{{.*}} : !fir.ref<i32>
! CHECK: arith.cmpi eq
! CHECK: cf.cond_br

! The NYI for unstructured loops associated with an OpenACC loop/combined
! directive only applies when the loop will be lowered as `independent`. The
! tests below exercise the relaxed cases where the user has not promised
! parallelism (seq/auto), so the lowering is expected to emit an `acc.loop`
! with explicit unstructured CFG inside.

! Combined `acc serial loop` (loop is `seq` by default) with STOP in body.
subroutine test_unstructured_serial_loop_stop(a)
  integer :: i, j
  real :: a(:,:,:)
  !$acc serial loop
  do i = 1, 10
    do j = 1, 10
      if (a(1,2,3) > 10.0) stop 'unstructured'
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured_serial_loop_stop
! CHECK: acc.serial combined(loop)
! CHECK: acc.loop combined(serial)
! CHECK: fir.call @_FortranAStopStatementText
! CHECK: } attributes {{{.*}}seq = [#acc.device_type<none>], unstructured}

! Standalone `acc loop seq` with STOP in body (explicit `seq` clause).
subroutine test_unstructured_loop_seq_stop(a)
  integer :: i, j
  real :: a(:,:,:)
  !$acc loop seq
  do i = 1, 10
    do j = 1, 10
      if (a(1,2,3) > 10.0) stop 'unstructured'
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured_loop_seq_stop
! CHECK: acc.loop private({{.*}})
! CHECK: fir.call @_FortranAStopStatementText
! CHECK: } attributes {{{.*}}seq = [#acc.device_type<none>], unstructured}

! Standalone `acc loop auto` with STOP in body (explicit `auto` clause).
subroutine test_unstructured_loop_auto_stop(a)
  integer :: i, j
  real :: a(:,:,:)
  !$acc loop auto
  do i = 1, 10
    do j = 1, 10
      if (a(1,2,3) > 10.0) stop 'unstructured'
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured_loop_auto_stop
! CHECK: acc.loop private({{.*}})
! CHECK: fir.call @_FortranAStopStatementText
! CHECK: } attributes {auto_ = [#acc.device_type<none>], {{.*}}unstructured}

! Standalone `acc loop` inside `acc serial` with STOP in body (loop is `seq`
! by default because parent compute construct is serial).
subroutine test_unstructured_loop_in_serial_stop(a)
  integer :: i, j
  real :: a(:,:,:)
  !$acc serial
  !$acc loop
  do i = 1, 10
    do j = 1, 10
      if (a(1,2,3) > 10.0) stop 'unstructured'
    end do
  end do
  !$acc end serial
end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured_loop_in_serial_stop
! CHECK: acc.serial
! CHECK: acc.loop private({{.*}})
! CHECK: fir.call @_FortranAStopStatementText
! CHECK: } attributes {{{.*}}seq = [#acc.device_type<none>], unstructured}

! Orphan `acc loop` inside a `seq` acc routine: loop is `seq` by default.
subroutine test_unstructured_orphan_loop_in_seq_routine(a)
  integer :: i, j
  real :: a(:,:,:)
  !$acc routine seq
  !$acc loop
  do i = 1, 10
    do j = 1, 10
      if (a(1,2,3) > 10.0) stop 'unstructured'
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured_orphan_loop_in_seq_routine
! CHECK: acc.loop private({{.*}})
! CHECK: fir.call @_FortranAStopStatementText
! CHECK: } attributes {{{.*}}seq = [#acc.device_type<none>], unstructured}

