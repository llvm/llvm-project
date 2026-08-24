! REQUIRES: openmp_runtime

! RUN: %flang_fc1 -emit-hlfir %openmp_flags %s -o - 2>&1 | FileCheck %s
! RUN: bbc %openmp_flags -emit-hlfir -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-fir %openmp_flags %s -o - 2>&1 | FileCheck %s
! RUN: bbc -emit-fir %openmp_flags -o - %s 2>&1 | FileCheck %s
!
! Test that omp_lib integer-kind wrapper procedures lower to module
! procedures instead of failing intrinsic resolution.

program main
  use omp_lib
  integer(1) :: i1 = 1_1
  integer(2) :: i2 = 2_2
  integer(4) :: i4 = 4_4
  integer(8) :: i8 = 8_8
  integer(omp_integer_kind) :: ires
  integer(omp_sched_kind) :: sched_kind
  integer(kmp_affinity_mask_kind) :: mask

  call omp_set_num_threads(i1)
  call omp_set_num_threads(i2)
  call omp_set_num_threads(i4)
  call omp_set_num_threads(i8)

  call omp_set_max_active_levels(i1)
  call omp_set_max_active_levels(i2)
  call omp_set_max_active_levels(i4)
  call omp_set_max_active_levels(i8)

  call omp_set_default_device(i1)
  call omp_set_default_device(i2)
  call omp_set_default_device(i4)
  call omp_set_default_device(i8)

  call omp_set_num_teams(i1)
  call omp_set_num_teams(i2)
  call omp_set_num_teams(i4)
  call omp_set_num_teams(i8)

  call omp_set_teams_thread_limit(i1)
  call omp_set_teams_thread_limit(i2)
  call omp_set_teams_thread_limit(i4)
  call omp_set_teams_thread_limit(i8)

  call kmp_set_stacksize(i1)
  call kmp_set_stacksize(i2)
  call kmp_set_stacksize(i4)
  call kmp_set_stacksize(i8)

  call kmp_set_blocktime(i1)
  call kmp_set_blocktime(i2)
  call kmp_set_blocktime(i4)
  call kmp_set_blocktime(i8)

  call kmp_set_library(i1)
  call kmp_set_library(i2)
  call kmp_set_library(i4)
  call kmp_set_library(i8)

  call kmp_set_disp_num_buffers(i1)
  call kmp_set_disp_num_buffers(i2)
  call kmp_set_disp_num_buffers(i4)
  call kmp_set_disp_num_buffers(i8)

  ires = omp_get_ancestor_thread_num(i1)
  ires = omp_get_ancestor_thread_num(i2)
  ires = omp_get_ancestor_thread_num(i4)
  ires = omp_get_ancestor_thread_num(i8)

  ires = omp_get_team_size(i1)
  ires = omp_get_team_size(i2)
  ires = omp_get_team_size(i4)
  ires = omp_get_team_size(i8)

  ires = omp_get_place_num_procs(i1)
  ires = omp_get_place_num_procs(i2)
  ires = omp_get_place_num_procs(i4)
  ires = omp_get_place_num_procs(i8)

  call omp_set_schedule(omp_sched_static, i1)
  call omp_set_schedule(omp_sched_static, i2)
  call omp_set_schedule(omp_sched_static, i4)
  call omp_set_schedule(omp_sched_static, i8)

  call omp_get_schedule(sched_kind, i1)
  call omp_get_schedule(sched_kind, i2)
  call omp_get_schedule(sched_kind, i4)
  call omp_get_schedule(sched_kind, i8)

  ires = omp_pause_resource(omp_pause_soft, i1)
  ires = omp_pause_resource(omp_pause_soft, i2)
  ires = omp_pause_resource(omp_pause_soft, i4)
  ires = omp_pause_resource(omp_pause_soft, i8)

  call kmp_create_affinity_mask(mask)
  ires = kmp_set_affinity_mask_proc(i1, mask)
  ires = kmp_set_affinity_mask_proc(i2, mask)
  ires = kmp_set_affinity_mask_proc(i4, mask)
  ires = kmp_set_affinity_mask_proc(i8, mask)

  ires = kmp_unset_affinity_mask_proc(i1, mask)
  ires = kmp_unset_affinity_mask_proc(i2, mask)
  ires = kmp_unset_affinity_mask_proc(i4, mask)
  ires = kmp_unset_affinity_mask_proc(i8, mask)

  ires = kmp_get_affinity_mask_proc(i1, mask)
  ires = kmp_get_affinity_mask_proc(i2, mask)
  ires = kmp_get_affinity_mask_proc(i4, mask)
  ires = kmp_get_affinity_mask_proc(i8, mask)

  block
    integer(4) :: ids4(1)
    integer(8) :: ids8(1)
    integer(4) :: parts4(1)
    integer(8) :: parts8(1)
    call omp_get_place_proc_ids(i4, ids4)
    call omp_get_place_proc_ids(i8, ids8)
    call omp_get_partition_place_nums(parts4)
    call omp_get_partition_place_nums(parts8)
  end block
end program

!CHECK-NOT: not yet implemented: intrinsic: omp_set_max_active_levels
!CHECK-NOT: not yet implemented: intrinsic: omp_set_default_device
!CHECK-NOT: not yet implemented: intrinsic: omp_set_num_teams
!CHECK-NOT: not yet implemented: intrinsic: omp_set_teams_thread_limit
!CHECK-NOT: not yet implemented: intrinsic: kmp_set_stacksize
!CHECK-NOT: not yet implemented: intrinsic: kmp_set_blocktime
!CHECK-NOT: not yet implemented: intrinsic: kmp_set_library
!CHECK-NOT: not yet implemented: intrinsic: kmp_set_disp_num_buffers
!CHECK-NOT: not yet implemented: intrinsic: omp_get_ancestor_thread_num
!CHECK-NOT: not yet implemented: intrinsic: omp_get_team_size
!CHECK-NOT: not yet implemented: intrinsic: omp_get_place_num_procs
!CHECK-NOT: not yet implemented: intrinsic: omp_set_schedule
!CHECK-NOT: not yet implemented: intrinsic: omp_get_schedule
!CHECK-NOT: not yet implemented: intrinsic: omp_pause_resource
!CHECK-NOT: not yet implemented: intrinsic: kmp_set_affinity_mask_proc
!CHECK-NOT: not yet implemented: intrinsic: kmp_unset_affinity_mask_proc
!CHECK-NOT: not yet implemented: intrinsic: kmp_get_affinity_mask_proc
!CHECK-NOT: not yet implemented: intrinsic: omp_get_place_proc_ids
!CHECK-NOT: not yet implemented: intrinsic: omp_get_partition_place_nums

!CHECK: fir.call @_QMomp_libPomp_set_num_threads_i1
!CHECK: fir.call @_QMomp_libPomp_set_num_threads_i2
!CHECK: fir.call @_QMomp_libPomp_set_num_threads_i4
!CHECK: fir.call @_QMomp_libPomp_set_num_threads_i8

!CHECK: fir.call @_QMomp_libPomp_set_max_active_levels_i1
!CHECK: fir.call @_QMomp_libPomp_set_max_active_levels_i2
!CHECK: fir.call @_QMomp_libPomp_set_max_active_levels_i4
!CHECK: fir.call @_QMomp_libPomp_set_max_active_levels_i8

!CHECK: fir.call @_QMomp_libPomp_set_default_device_i1
!CHECK: fir.call @_QMomp_libPomp_set_default_device_i2
!CHECK: fir.call @_QMomp_libPomp_set_default_device_i4
!CHECK: fir.call @_QMomp_libPomp_set_default_device_i8

!CHECK: fir.call @_QMomp_libPomp_set_num_teams_i1
!CHECK: fir.call @_QMomp_libPomp_set_num_teams_i2
!CHECK: fir.call @_QMomp_libPomp_set_num_teams_i4
!CHECK: fir.call @_QMomp_libPomp_set_num_teams_i8

!CHECK: fir.call @_QMomp_libPomp_set_teams_thread_limit_i1
!CHECK: fir.call @_QMomp_libPomp_set_teams_thread_limit_i2
!CHECK: fir.call @_QMomp_libPomp_set_teams_thread_limit_i4
!CHECK: fir.call @_QMomp_libPomp_set_teams_thread_limit_i8

!CHECK: fir.call @_QMomp_libPkmp_set_stacksize_i1
!CHECK: fir.call @_QMomp_libPkmp_set_stacksize_i2
!CHECK: fir.call @_QMomp_libPkmp_set_stacksize_i4
!CHECK: fir.call @_QMomp_libPkmp_set_stacksize_i8

!CHECK: fir.call @_QMomp_libPkmp_set_blocktime_i1
!CHECK: fir.call @_QMomp_libPkmp_set_blocktime_i2
!CHECK: fir.call @_QMomp_libPkmp_set_blocktime_i4
!CHECK: fir.call @_QMomp_libPkmp_set_blocktime_i8

!CHECK: fir.call @_QMomp_libPkmp_set_library_i1
!CHECK: fir.call @_QMomp_libPkmp_set_library_i2
!CHECK: fir.call @_QMomp_libPkmp_set_library_i4
!CHECK: fir.call @_QMomp_libPkmp_set_library_i8

!CHECK: fir.call @_QMomp_libPkmp_set_disp_num_buffers_i1
!CHECK: fir.call @_QMomp_libPkmp_set_disp_num_buffers_i2
!CHECK: fir.call @_QMomp_libPkmp_set_disp_num_buffers_i4
!CHECK: fir.call @_QMomp_libPkmp_set_disp_num_buffers_i8

!CHECK: fir.call @_QMomp_libPomp_get_ancestor_thread_num_i1
!CHECK: fir.call @_QMomp_libPomp_get_ancestor_thread_num_i2
!CHECK: fir.call @_QMomp_libPomp_get_ancestor_thread_num_i4
!CHECK: fir.call @_QMomp_libPomp_get_ancestor_thread_num_i8

!CHECK: fir.call @_QMomp_libPomp_get_team_size_i1
!CHECK: fir.call @_QMomp_libPomp_get_team_size_i2
!CHECK: fir.call @_QMomp_libPomp_get_team_size_i4
!CHECK: fir.call @_QMomp_libPomp_get_team_size_i8

!CHECK: fir.call @_QMomp_libPomp_get_place_num_procs_i1
!CHECK: fir.call @_QMomp_libPomp_get_place_num_procs_i2
!CHECK: fir.call @_QMomp_libPomp_get_place_num_procs_i4
!CHECK: fir.call @_QMomp_libPomp_get_place_num_procs_i8

!CHECK: fir.call @_QMomp_libPomp_set_schedule_i1
!CHECK: fir.call @_QMomp_libPomp_set_schedule_i2
!CHECK: fir.call @_QMomp_libPomp_set_schedule_i4
!CHECK: fir.call @_QMomp_libPomp_set_schedule_i8

!CHECK: fir.call @_QMomp_libPomp_get_schedule_i1
!CHECK: fir.call @_QMomp_libPomp_get_schedule_i2
!CHECK: fir.call @_QMomp_libPomp_get_schedule_i4
!CHECK: fir.call @_QMomp_libPomp_get_schedule_i8

!CHECK: fir.call @_QMomp_libPomp_pause_resource_i1
!CHECK: fir.call @_QMomp_libPomp_pause_resource_i2
!CHECK: fir.call @_QMomp_libPomp_pause_resource_i4
!CHECK: fir.call @_QMomp_libPomp_pause_resource_i8

!CHECK: fir.call @_QMomp_libPkmp_set_affinity_mask_proc_i1
!CHECK: fir.call @_QMomp_libPkmp_set_affinity_mask_proc_i2
!CHECK: fir.call @_QMomp_libPkmp_set_affinity_mask_proc_i4
!CHECK: fir.call @_QMomp_libPkmp_set_affinity_mask_proc_i8

!CHECK: fir.call @_QMomp_libPkmp_unset_affinity_mask_proc_i1
!CHECK: fir.call @_QMomp_libPkmp_unset_affinity_mask_proc_i2
!CHECK: fir.call @_QMomp_libPkmp_unset_affinity_mask_proc_i4
!CHECK: fir.call @_QMomp_libPkmp_unset_affinity_mask_proc_i8

!CHECK: fir.call @_QMomp_libPkmp_get_affinity_mask_proc_i1
!CHECK: fir.call @_QMomp_libPkmp_get_affinity_mask_proc_i2
!CHECK: fir.call @_QMomp_libPkmp_get_affinity_mask_proc_i4
!CHECK: fir.call @_QMomp_libPkmp_get_affinity_mask_proc_i8

!CHECK: fir.call @_QMomp_libPomp_get_place_proc_ids_i4
!CHECK: fir.call @_QMomp_libPomp_get_place_proc_ids_i8
!CHECK: fir.call @_QMomp_libPomp_get_partition_place_nums_i4
!CHECK: fir.call @_QMomp_libPomp_get_partition_place_nums_i8
