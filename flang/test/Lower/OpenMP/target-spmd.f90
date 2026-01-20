! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPdistribute_parallel_do_generic() {
subroutine distribute_parallel_do_generic()
  ! CHECK: omp.target
  ! CHECK-NOT: host_eval({{.*}})
  ! CHECK-SAME: {
  !$omp target
  !$omp teams
  !$omp distribute parallel do
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute parallel do
  call bar() !< Prevents this from being SPMD.
  !$omp end teams
  !$omp end target

  ! CHECK: omp.target
  ! CHECK-NOT: host_eval({{.*}})
  ! CHECK-SAME: {
  !$omp target teams
  !$omp distribute parallel do
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute parallel do
  call bar() !< Prevents this from being SPMD.
  !$omp end target teams

  ! CHECK: omp.target
  ! CHECK-NOT: host_eval({{.*}})
  ! CHECK-SAME: {
  !$omp target teams
  !$omp distribute parallel do
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute parallel do

  !$omp distribute parallel do
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute parallel do
  !$omp end target teams
end subroutine distribute_parallel_do_generic

! CHECK-LABEL: func.func @_QPdistribute_parallel_do_spmd() {
subroutine distribute_parallel_do_spmd()
  ! CHECK: omp.target
  ! CHECK-SAME: host_eval({{.*}})
  !$omp target
  !$omp teams
  !$omp distribute parallel do
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute parallel do
  !$omp end teams
  !$omp end target

  ! CHECK: omp.target
  ! CHECK-SAME: host_eval({{.*}})
  !$omp target teams
  !$omp distribute parallel do
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute parallel do
  !$omp end target teams
end subroutine distribute_parallel_do_spmd

! CHECK-LABEL: func.func @_QPdistribute_parallel_do_simd_generic() {
subroutine distribute_parallel_do_simd_generic()
  ! CHECK: omp.target
  ! CHECK-NOT: host_eval({{.*}})
  ! CHECK-SAME: {
  !$omp target
  !$omp teams
  !$omp distribute parallel do simd
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute parallel do simd
  call bar() !< Prevents this from being SPMD.
  !$omp end teams
  !$omp end target

  ! CHECK: omp.target
  ! CHECK-NOT: host_eval({{.*}})
  ! CHECK-SAME: {
  !$omp target teams
  !$omp distribute parallel do simd
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute parallel do simd
  call bar() !< Prevents this from being SPMD.
  !$omp end target teams

  ! CHECK: omp.target
  ! CHECK-NOT: host_eval({{.*}})
  ! CHECK-SAME: {
  !$omp target teams
  !$omp distribute parallel do simd
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute parallel do simd

  !$omp distribute parallel do simd
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute parallel do simd
  !$omp end target teams
end subroutine distribute_parallel_do_simd_generic

! CHECK-LABEL: func.func @_QPdistribute_parallel_do_simd_spmd() {
subroutine distribute_parallel_do_simd_spmd()
  ! CHECK: omp.target
  ! CHECK-SAME: host_eval({{.*}})
  !$omp target
  !$omp teams
  !$omp distribute parallel do simd
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute parallel do simd
  !$omp end teams
  !$omp end target

  ! CHECK: omp.target
  ! CHECK-SAME: host_eval({{.*}})
  !$omp target teams
  !$omp distribute parallel do simd
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute parallel do simd
  !$omp end target teams
end subroutine distribute_parallel_do_simd_spmd

! CHECK-LABEL: func.func @_QPteams_distribute_parallel_do_spmd() {
subroutine teams_distribute_parallel_do_spmd()
  ! CHECK: omp.target
  ! CHECK-SAME: host_eval({{.*}})
  !$omp target
  !$omp teams distribute parallel do
  do i = 1, 10
    call foo(i)
  end do
  !$omp end teams distribute parallel do
  !$omp end target
end subroutine teams_distribute_parallel_do_spmd

! CHECK-LABEL: func.func @_QPteams_distribute_parallel_do_simd_spmd() {
subroutine teams_distribute_parallel_do_simd_spmd()
  ! CHECK: omp.target
  ! CHECK-SAME: host_eval({{.*}})
  !$omp target
  !$omp teams distribute parallel do simd
  do i = 1, 10
    call foo(i)
  end do
  !$omp end teams distribute parallel do simd
  !$omp end target
end subroutine teams_distribute_parallel_do_simd_spmd

! CHECK-LABEL: func.func @_QPtarget_teams_distribute_parallel_do_spmd() {
subroutine target_teams_distribute_parallel_do_spmd()
  ! CHECK: omp.target
  ! CHECK-SAME: host_eval({{.*}})
  !$omp target teams distribute parallel do
  do i = 1, 10
    call foo(i)
  end do
  !$omp end target teams distribute parallel do
end subroutine target_teams_distribute_parallel_do_spmd

! CHECK-LABEL: func.func @_QPtarget_teams_distribute_parallel_do_simd_spmd() {
subroutine target_teams_distribute_parallel_do_simd_spmd()
  ! CHECK: omp.target
  ! CHECK-SAME: host_eval({{.*}})
  !$omp target teams distribute parallel do simd
  do i = 1, 10
    call foo(i)
  end do
  !$omp end target teams distribute parallel do simd
end subroutine target_teams_distribute_parallel_do_simd_spmd

!===============================================================================
! Target teams `device` clause
!===============================================================================

!CHECK-LABEL: func.func @_QPomp_target_teams_device() {
subroutine omp_target_teams_device
  integer            :: dev32
  integer(kind=8)    :: dev64
  integer(kind=2)    :: dev16

  dev32 = 1
  dev64 = 2_8
  dev16 = 3_2

  !$omp target teams device(dev32)
  !$omp end target teams
  ! CHECK: %[[DEV32:.*]] = fir.load %{{.*}} : !fir.ref<i32>
  ! CHECK: omp.target device(%[[DEV32]] : i32)

  !$omp target teams device(dev64)
  !$omp end target teams
  ! CHECK: %[[DEV64:.*]] = fir.load %{{.*}} : !fir.ref<i64>
  ! CHECK: omp.target device(%[[DEV64]] : i64)

  !$omp target teams device(dev16)
  !$omp end target teams
  ! CHECK: %[[DEV16:.*]] = fir.load %{{.*}} : !fir.ref<i16>
  ! CHECK: omp.target device(%[[DEV16]] : i16)

  !$omp target teams device(2)
  !$omp end target teams
  ! CHECK: %[[C2:.*]] = arith.constant 2 : i32
  ! CHECK: omp.target device(%[[C2]] : i32)

  !$omp target teams device(5_8)
  !$omp end target teams
  ! CHECK: %[[C5:.*]] = arith.constant 5 : i64
  ! CHECK: omp.target device(%[[C5]] : i64)

end subroutine omp_target_teams_device

!===============================================================================
! Target teams distribute `device` clause
!===============================================================================

!CHECK-LABEL: func.func @_QPomp_target_teams_distribute_device() {
subroutine omp_target_teams_distribute_device
  integer            :: dev32
  integer(kind=8)    :: dev64
  integer(kind=2)    :: dev16
  integer            :: i

  dev32 = 1
  dev64 = 2_8
  dev16 = 3_2

  !$omp target teams distribute device(dev32)
  do i = 1, 1
  end do
  !$omp end target teams distribute
  ! CHECK: %[[DEV32:.*]] = fir.load %{{.*}} : !fir.ref<i32>
  ! CHECK: omp.target device(%[[DEV32]] : i32)
  ! CHECK: omp.teams
  ! CHECK: omp.distribute
  ! CHECK: omp.loop_nest

  !$omp target teams distribute device(dev64)
  do i = 1, 1
  end do
  !$omp end target teams distribute
  ! CHECK: %[[DEV64:.*]] = fir.load %{{.*}} : !fir.ref<i64>
  ! CHECK: omp.target device(%[[DEV64]] : i64)

  !$omp target teams distribute device(dev16)
  do i = 1, 1
  end do
  !$omp end target teams distribute
  ! CHECK: %[[DEV16:.*]] = fir.load %{{.*}} : !fir.ref<i16>
  ! CHECK: omp.target device(%[[DEV16]] : i16)

  !$omp target teams distribute device(2)
  do i = 1, 1
  end do
  !$omp end target teams distribute
  ! CHECK: %[[C2:.*]] = arith.constant 2 : i32
  ! CHECK: omp.target device(%[[C2]] : i32)

  !$omp target teams distribute device(5_8)
  do i = 1, 1
  end do
  !$omp end target teams distribute
  ! CHECK: %[[C5:.*]] = arith.constant 5 : i64
  ! CHECK: omp.target device(%[[C5]] : i64)

end subroutine omp_target_teams_distribute_device

!===============================================================================
! Target teams distribute parallel loop `device` clause
!===============================================================================

!CHECK-LABEL: func.func @_QPomp_target_teams_distribute_parallel_do_device() {
subroutine omp_target_teams_distribute_parallel_do_device
  integer            :: dev32
  integer(kind=8)    :: dev64
  integer(kind=2)    :: dev16
  integer            :: i

  dev32 = 1
  dev64 = 2_8
  dev16 = 3_2

  !$omp target teams distribute parallel do device(dev32)
  do i = 1, 1
  end do
  !$omp end target teams distribute parallel do
  ! CHECK: %[[DEV32:.*]] = fir.load %{{.*}} : !fir.ref<i32>
  ! CHECK: omp.target device(%[[DEV32]] : i32)
  ! CHECK: omp.teams
  ! CHECK: omp.parallel
  ! CHECK: omp.distribute
  ! CHECK: omp.wsloop
  ! CHECK: omp.loop_nest

  !$omp target teams distribute parallel do device(dev64)
  do i = 1, 1
  end do
  !$omp end target teams distribute parallel do
  ! CHECK: %[[DEV64:.*]] = fir.load %{{.*}} : !fir.ref<i64>
  ! CHECK: omp.target device(%[[DEV64]] : i64)

  !$omp target teams distribute parallel do device(dev16)
  do i = 1, 1
  end do
  !$omp end target teams distribute parallel do
  ! CHECK: %[[DEV16:.*]] = fir.load %{{.*}} : !fir.ref<i16>
  ! CHECK: omp.target device(%[[DEV16]] : i16)

  !$omp target teams distribute parallel do device(2)
  do i = 1, 1
  end do
  !$omp end target teams distribute parallel do
  ! CHECK: %[[C2:.*]] = arith.constant 2 : i32
  ! CHECK: omp.target device(%[[C2]] : i32)

  !$omp target teams distribute parallel do device(5_8)
  do i = 1, 1
  end do
  !$omp end target teams distribute parallel do
  ! CHECK: %[[C5:.*]] = arith.constant 5 : i64
  ! CHECK: omp.target device(%[[C5]] : i64)

end subroutine omp_target_teams_distribute_parallel_do_device

!===============================================================================
! Target teams distribute parallel loop simd `device` clause
!===============================================================================

!CHECK-LABEL: func.func @_QPomp_target_teams_distribute_parallel_do_simd_device() {
subroutine omp_target_teams_distribute_parallel_do_simd_device
  integer            :: dev32
  integer(kind=8)    :: dev64
  integer(kind=2)    :: dev16
  integer            :: i

  dev32 = 1
  dev64 = 2_8
  dev16 = 3_2

  !$omp target teams distribute parallel do simd device(dev32)
  do i = 1, 1
  end do
  !$omp end target teams distribute parallel do simd
  ! CHECK: %[[DEV32:.*]] = fir.load %{{.*}} : !fir.ref<i32>
  ! CHECK: omp.target device(%[[DEV32]] : i32)
  ! CHECK: omp.teams
  ! CHECK: omp.parallel
  ! CHECK: omp.distribute
  ! CHECK: omp.wsloop
  ! CHECK: omp.simd
  ! CHECK: omp.loop_nest

  !$omp target teams distribute parallel do simd device(dev64)
  do i = 1, 1
  end do
  !$omp end target teams distribute parallel do simd
  ! CHECK: %[[DEV64:.*]] = fir.load %{{.*}} : !fir.ref<i64>
  ! CHECK: omp.target device(%[[DEV64]] : i64)

  !$omp target teams distribute parallel do simd device(dev16)
  do i = 1, 1
  end do
  !$omp end target teams distribute parallel do simd
  ! CHECK: %[[DEV16:.*]] = fir.load %{{.*}} : !fir.ref<i16>
  ! CHECK: omp.target device(%[[DEV16]] : i16)

  !$omp target teams distribute parallel do simd device(2)
  do i = 1, 1
  end do
  !$omp end target teams distribute parallel do simd
  ! CHECK: %[[C2:.*]] = arith.constant 2 : i32
  ! CHECK: omp.target device(%[[C2]] : i32)

  !$omp target teams distribute parallel do simd device(5_8)
  do i = 1, 1
  end do
  !$omp end target teams distribute parallel do simd
  ! CHECK: %[[C5:.*]] = arith.constant 5 : i64
  ! CHECK: omp.target device(%[[C5]] : i64)

end subroutine omp_target_teams_distribute_parallel_do_simd_device
