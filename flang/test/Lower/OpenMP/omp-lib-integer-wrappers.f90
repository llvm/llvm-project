! REQUIRES: openmp_runtime

! RUN: %flang_fc1 -emit-hlfir %openmp_flags %s -o - 2>&1 | FileCheck %s
! RUN: bbc %openmp_flags -emit-hlfir -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-fir %openmp_flags %s -o - 2>&1 | FileCheck %s
! RUN: bbc -emit-fir %openmp_flags -o - %s 2>&1 | FileCheck %s
!
! Test that omp_lib generic interfaces select the default bind(c) entry for
! omp_integer_kind arguments and the kmp_*_8 entries for integer(8).

program main
  use omp_lib
  integer(omp_integer_kind) :: i4
  integer(8) :: i8
  integer(omp_integer_kind) :: ires
  integer(omp_sched_kind) :: sched_kind
  integer(kmp_affinity_mask_kind) :: mask
  integer(omp_integer_kind) :: ids4(1)
  integer(8) :: ids8(1)
  integer(omp_integer_kind) :: parts4(1)
  integer(8) :: parts8(1)

  i4 = 4
  i8 = 8_8

  call omp_set_num_threads(i4)
  call omp_set_num_threads(i8)

  call omp_set_max_active_levels(i4)
  call omp_set_max_active_levels(i8)

  call omp_set_default_device(i4)
  call omp_set_default_device(i8)

  call omp_set_num_teams(i4)
  call omp_set_num_teams(i8)

  call omp_set_teams_thread_limit(i4)
  call omp_set_teams_thread_limit(i8)

  call kmp_set_stacksize(i4)
  call kmp_set_stacksize(i8)

  call kmp_set_blocktime(i4)
  call kmp_set_blocktime(i8)

  call kmp_set_library(i4)
  call kmp_set_library(i8)

  call kmp_set_disp_num_buffers(i4)
  call kmp_set_disp_num_buffers(i8)

  ires = omp_get_ancestor_thread_num(i4)
  ires = omp_get_ancestor_thread_num(i8)

  ires = omp_get_team_size(i4)
  ires = omp_get_team_size(i8)

  ires = omp_get_place_num_procs(i4)
  ires = omp_get_place_num_procs(i8)

  call omp_set_schedule(omp_sched_static, i4)
  call omp_set_schedule(omp_sched_static, i8)

  call omp_get_schedule(sched_kind, i4)
  call omp_get_schedule(sched_kind, i8)

  ires = omp_pause_resource(omp_pause_soft, i4)
  ires = omp_pause_resource(omp_pause_soft, i8)

  call kmp_create_affinity_mask(mask)
  ires = kmp_set_affinity_mask_proc(i4, mask)
  ires = kmp_set_affinity_mask_proc(i8, mask)

  ires = kmp_unset_affinity_mask_proc(i4, mask)
  ires = kmp_unset_affinity_mask_proc(i8, mask)

  ires = kmp_get_affinity_mask_proc(i4, mask)
  ires = kmp_get_affinity_mask_proc(i8, mask)

  call omp_get_place_proc_ids(i4, ids4)
  call omp_get_place_proc_ids(i8, ids8)

  call omp_get_partition_place_nums(parts4)
  call omp_get_partition_place_nums(parts8)
end program

! CHECK-NOT: not yet implemented: intrinsic:

! CHECK: fir.call @omp_set_num_threads(
! CHECK: fir.call @kmp_set_num_threads_8(

! CHECK: fir.call @omp_set_max_active_levels(
! CHECK: fir.call @kmp_set_max_active_levels_8(

! CHECK: fir.call @omp_set_default_device(
! CHECK: fir.call @kmp_set_default_device_8(

! CHECK: fir.call @omp_set_num_teams(
! CHECK: fir.call @kmp_set_num_teams_8(

! CHECK: fir.call @omp_set_teams_thread_limit(
! CHECK: fir.call @kmp_set_teams_thread_limit_8(

! CHECK: fir.call @kmp_set_stacksize(
! CHECK: fir.call @kmp_set_stacksize_8(

! CHECK: fir.call @kmp_set_blocktime(
! CHECK: fir.call @kmp_set_blocktime_8(

! CHECK: fir.call @kmp_set_library(
! CHECK: fir.call @kmp_set_library_8(

! CHECK: fir.call @kmp_set_disp_num_buffers(
! CHECK: fir.call @kmp_set_disp_num_buffers_8(

! CHECK: fir.call @omp_get_ancestor_thread_num(
! CHECK: fir.call @kmp_get_ancestor_thread_num_8(

! CHECK: fir.call @omp_get_team_size(
! CHECK: fir.call @kmp_get_team_size_8(

! CHECK: fir.call @omp_get_place_num_procs(
! CHECK: fir.call @kmp_get_place_num_procs_8(

! CHECK: fir.call @omp_set_schedule(
! CHECK: fir.call @kmp_set_schedule_8(

! CHECK: fir.call @omp_get_schedule(
! CHECK: fir.call @kmp_get_schedule_8(

! CHECK: fir.call @omp_pause_resource(
! CHECK: fir.call @kmp_pause_resource_8(

! CHECK: fir.call @kmp_set_affinity_mask_proc(
! CHECK: fir.call @kmp_set_affinity_mask_proc_8(

! CHECK: fir.call @kmp_unset_affinity_mask_proc(
! CHECK: fir.call @kmp_unset_affinity_mask_proc_8(

! CHECK: fir.call @kmp_get_affinity_mask_proc(
! CHECK: fir.call @kmp_get_affinity_mask_proc_8(

! CHECK: fir.call @omp_get_place_proc_ids(
! CHECK: fir.call @kmp_get_place_proc_ids_8(

! CHECK: fir.call @omp_get_partition_place_nums(
! CHECK: fir.call @kmp_get_partition_place_nums_8(
