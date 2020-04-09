!===-- module/omp_lib.h ------------------------------------------*- F90 -*-===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

!dir$ free

  integer, parameter :: omp_integer_kind = selected_int_kind(9) ! 32-bit int
  integer, parameter :: omp_logical_kind = kind(.true.)

  integer, parameter :: omp_sched_kind = omp_integer_kind
  integer, parameter :: omp_proc_bind_kind = omp_integer_kind
  integer, parameter :: omp_pause_resource_kind = omp_integer_kind
  integer, parameter :: omp_sync_hint_kind = omp_integer_kind
  integer, parameter :: omp_lock_hint_kind = omp_sync_hint_kind
  integer, parameter :: omp_event_handle_kind = omp_integer_kind
  integer, parameter :: omp_alloctrait_key_kind = omp_integer_kind
  integer, parameter :: omp_alloctrait_val_kind = omp_integer_kind
  integer, parameter :: omp_allocator_handle_kind = omp_integer_kind
  integer, parameter :: omp_memspace_handle_kind = omp_integer_kind
  integer, parameter :: omp_lock_kind = int_ptr_kind()
  integer, parameter :: omp_nest_lock_kind = int_ptr_kind()
  integer, parameter :: omp_depend_kind = omp_integer_kind

  integer(kind=omp_sched_kind), parameter :: &
    omp_sched_static = 1, &
    omp_sched_dynamic = 2, &
    omp_sched_guided = 3, &
    omp_sched_auto = 4

  integer(kind=omp_proc_bind_kind), parameter :: &
    omp_proc_bind_false = 0, &
    omp_proc_bind_true = 1, &
    omp_proc_bind_master = 2, &
    omp_proc_bind_close = 3, &
    omp_proc_bind_spread = 4

  integer(kind=omp_pause_resource_kind), parameter :: &
    omp_pause_soft = 1, &
    omp_pause_hard = 2

  integer(kind=omp_sync_hint_kind), parameter :: &
    omp_sync_hint_none = 0, &
    omp_sync_hint_uncontended = 1, &
    omp_sync_hint_contended = 2, &
    omp_sync_hint_nonspeculative = 4, &
    omp_sync_hint_speculative = 8
  integer(kind=omp_lock_hint_kind), parameter :: &
    omp_lock_hint_none = omp_sync_hint_none, &
    omp_lock_hint_uncontended = omp_sync_hint_uncontended, &
    omp_lock_hint_contended = omp_sync_hint_contended, &
    omp_lock_hint_nonspeculative = omp_sync_hint_nonspeculative, &
    omp_lock_hint_speculative = omp_sync_hint_speculative

  integer(kind=omp_event_handle_kind), parameter :: &
    omp_allow_completion_event = 0, &
    omp_task_fulfill_event = 1

  integer(kind=omp_alloctrait_key_kind), parameter :: &
    omp_atk_sync_hint = 1, &
    omp_atk_alignment = 2, &
    omp_atk_access = 3, &
    omp_atk_pool_size = 4, &
    omp_atk_fallback = 5, &
    omp_atk_fb_data = 6, &
    omp_atk_pinned = 7, &
    omp_atk_partition = 8

  integer(kind=omp_alloctrait_val_kind), parameter :: &
    omp_atv_false = 0, &
    omp_atv_true = 1, &
    omp_atv_default = 2, &
    omp_atv_contended = 3, &
    omp_atv_uncontended = 4, &
    omp_atv_sequential = 5, &
    omp_atv_private = 6, &
    omp_atv_all = 7, &
    omp_atv_thread = 8, &
    omp_atv_pteam = 9, &
    omp_atv_cgroup = 10, &
    omp_atv_default_mem_fb = 11, &
    omp_atv_null_fb = 12, &
    omp_atv_abort_fb = 13, &
    omp_atv_allocator_fb = 14, &
    omp_atv_environment = 15, &
    omp_atv_nearest = 16, &
    omp_atv_blocked = 17, &
    omp_atv_interleaved = 18

  type :: omp_alloctrait
    integer(kind=omp_alloctrait_key_kind) :: key, value
  end type omp_alloctrait

  integer(kind=omp_allocator_handle_kind), parameter :: omp_null_allocator = 0

  integer(kind=omp_memspace_handle_kind), parameter :: &
    omp_default_mem_space = 0, &
    omp_large_cap_mem_space = 0, &
    omp_const_mem_space = 0, &
    omp_high_bw_mem_space = 0, &
    omp_low_lat_mem_space = 0, &
    omp_default_mem_alloc = 1, &
    omp_large_cap_mem_alloc = omp_default_mem_alloc, &
    omp_const_mem_alloc = 1, &
    omp_high_bw_mem_alloc = 1, &
    omp_low_lat_mem_alloc = 1, &
    omp_thread_mem_alloc = omp_atv_thread, &
    omp_pteam_mem_alloc = omp_atv_pteam, &
    omp_cgroup_mem_alloc = omp_atv_cgroup

  integer(kind=omp_integer_kind), parameter :: openmp_version = 200805

  interface

    subroutine omp_set_num_threads(nthreads) bind(c)
      import
      integer(kind=omp_integer_kind), value :: nthreads
    end subroutine omp_set_num_threads

    function omp_get_num_threads() bind(c)
      import
      integer(kind=omp_integer_kind) :: omp_get_num_threads
    end function omp_get_num_threads

    function omp_get_max_threads() bind(c)
      import
      integer(kind=omp_integer_kind) :: omp_get_max_threads
    end function omp_get_max_threads

    function omp_get_thread_num() bind(c)
      import
      integer(kind=omp_integer_kind) :: omp_get_thread_num
    end function omp_get_thread_num

    function omp_get_num_procs() bind(c)
      import
      integer(kind=omp_integer_kind) :: omp_get_num_procs
    end function omp_get_num_procs

    function omp_in_parallel() bind(c)
      import
      logical(kind=omp_logical_kind) :: omp_in_parallel
    end function omp_in_parallel

    subroutine omp_set_dynamic(enable) bind(c)
      import
      logical(kind=omp_logical_kind), value :: enable
    end subroutine omp_set_dynamic

    function omp_get_dynamic() bind(c)
      import
      logical(kind=omp_logical_kind) :: omp_get_dynamic
    end function omp_get_dynamic

    function omp_get_cancelation() bind(c)
      import
      logical(kind=omp_logical_kind) :: omp_get_cancelation
    end function omp_get_cancelation

    subroutine omp_set_nested(enable) bind(c)
      import
      logical(kind=omp_logical_kind), value :: enable
    end subroutine omp_set_nested

    function omp_get_nested() bind(c)
      import
      logical(kind=omp_logical_kind) ::omp_get_nested
    end function omp_get_nested

    subroutine omp_set_schedule(kind, modifier) bind(c)
      import
      integer(kind=omp_integer_kind), value :: kind, modifier
    end subroutine omp_set_schedule

    subroutine omp_get_schedule(kind, modifier) bind(c)
      import
      integer(kind=omp_integer_kind), intent(out) :: kind, modifier
    end subroutine omp_get_schedule

    function omp_get_thread_limit() bind(c)
      import
      integer(kind=omp_integer_kind) :: omp_get_thread_limit
    end function omp_get_thread_limit

    function omp_get_supported_active_levels() bind(c)
      import
      integer(kind=omp_integer_kind) :: omp_get_supported_active_levels
    end function omp_get_supported_active_levels

    subroutine omp_set_max_active_levels(max_levels) bind(c)
      import
      integer(kind=omp_integer_kind), value :: max_levels
    end subroutine omp_set_max_active_levels

    function omp_get_max_active_levels() bind(c)
      import
      integer(kind=omp_integer_kind) :: omp_get_max_active_levels
    end function omp_get_max_active_levels

    function omp_get_level() bind(c)
      import
      integer(kind=omp_integer_kind) :: omp_get_level
    end function omp_get_level

    function omp_get_ancestor_thread_num(level) bind(c)
      import
      integer(kind=omp_integer_kind), value :: level
      integer(kind=omp_integer_kind) :: omp_get_ancestor_thread_num
    end function omp_get_ancestor_thread_num

    function omp_get_team_size(level) bind(c)
      import
      integer(kind=omp_integer_kind), value :: level
      integer(kind=omp_integer_kind) :: omp_get_team_size
    end function omp_get_team_size

    function omp_get_active_level() bind(c)
      import
      integer(kind=omp_integer_kind) :: omp_get_active_level
    end function omp_get_active_level

    function omp_in_final() bind(c)
      import
      logical(kind=omp_logical_kind) :: omp_in_final
    end function omp_in_final

    function omp_get_proc_bind() bind(c)
      import
      integer(kind=omp_proc_bind_kind) :: omp_get_proc_bind
    end function omp_get_proc_bind

    function omp_get_num_places() bind(c)
      import
      integer(kind=omp_integer_kind) :: omp_get_num_places
    end function omp_get_num_places

    function omp_get_place_num_procs(place_num) bind(c)
      import
      integer(kind=omp_integer_kind), value :: place_num
      integer(kind=omp_integer_kind) omp_get_place_num_procs
    end function omp_get_place_num_procs

    subroutine omp_get_place_proc_ids(place_num, ids) bind(c)
      import
      integer(kind=omp_integer_kind), value :: place_num
      integer(kind=omp_integer_kind), intent(out) :: ids(*)
    end subroutine omp_get_place_proc_ids

    function omp_get_place_num() bind(c)
      import
      integer(kind=omp_integer_kind) :: omp_get_place_num
    end function omp_get_place_num

    function omp_get_partition_num_places() bind(c)
      import
      integer(kind=omp_integer_kind) :: omp_get_partition_num_places
    end function omp_get_partition_num_places

    subroutine omp_get_partition_place_nums(place_nums) bind(c)
      import
      integer(kind=omp_integer_kind), intent(out) :: place_nums(*)
    end subroutine omp_get_partition_place_nums

    subroutine omp_set_affinity_format(format)
      import
      character(len=*), intent(in) :: format
    end subroutine omp_set_affinity_format

    function omp_get_affinity_format(buffer)
      import
      character(len=*), intent(out) :: buffer
      integer(kind=omp_integer_kind) :: omp_get_affinity_format
    end function omp_get_affinity_format

    subroutine omp_display_affinity(format)
      import
      character(len=*), intent(in) :: format
    end subroutine omp_display_affinity

    function omp_capture_affinity(buffer, format)
      import
      character(len=*), intent(out) :: buffer
      character(len=*), intent(in) :: format
      integer(kind=omp_integer_kind) omp_capture_affinity
    end function omp_capture_affinity

    subroutine omp_set_default_device(device_num) bind(c)
      import
      integer(kind=omp_integer_kind), value :: device_num
    end subroutine omp_set_default_device

    function omp_get_default_device() bind(c)
      import
      integer(kind=omp_integer_kind) :: omp_get_default_device
    end function omp_get_default_device

    function omp_get_num_devices() bind(c)
      import
      integer(kind=omp_integer_kind) :: omp_get_num_devices
    end function omp_get_num_devices

    function omp_get_device_num() bind(c)
      import
      integer(kind=omp_integer_kind) :: omp_get_device_num
    end function omp_get_device_num

    function omp_get_num_teams() bind(c)
      import
      integer(kind=omp_integer_kind) :: omp_get_num_teams
    end function omp_get_num_teams

    function omp_get_team_num() bind(c)
      import
      integer(kind=omp_integer_kind) :: omp_get_team_num
    end function omp_get_team_num

    function omp_is_initial_device() bind(c)
      import
      integer(kind=omp_logical_kind) :: omp_is_initial_device ! TODO: should this be LOGICAL?
    end function omp_is_initial_device

    function omp_get_initial_device() bind(c)
      import
      integer(kind=omp_integer_kind) :: omp_get_initial_device
    end function omp_get_initial_device

    function omp_get_max_task_priority() bind(c)
      import
      integer(kind=omp_integer_kind) :: omp_get_max_task_priority
    end function omp_get_max_task_priority

    function omp_pause_resource(kind, device_num) bind(c)
      import
      integer(kind=omp_pause_resource_kind), value :: kind
      integer(kind=omp_integer_kind), value :: device_num
      integer(kind=omp_integer_kind) :: omp_pause_resource
    end function omp_pause_resource

    function omp_pause_resource_all(kind)
      import
      integer(kind=omp_pause_resource_kind), value :: kind
      integer(kind=omp_integer_kind) :: omp_pause_resource_all
    end function omp_pause_resource_all

! Lock routines
    subroutine omp_init_lock(lockvar) bind(c, name="omp_init_lock_")
      import
      integer(kind=omp_lock_kind), intent(out) :: lockvar
    end subroutine omp_init_lock

    subroutine omp_init_lock_with_hint(lockvar, hint) bind(c, name="omp_init_lock_with_hint_")
      import
      integer(kind=omp_lock_kind), intent(out) :: lockvar
      integer(kind=omp_sync_hint_kind), value :: hint
    end subroutine omp_init_lock_with_hint

    subroutine omp_destroy_lock(lockvar) bind(c, name="omp_destroy_lock_")
      import
      integer(kind=omp_lock_kind), intent(inout) :: lockvar
    end subroutine omp_destroy_lock

    subroutine omp_set_lock(lockvar) bind(c, name="omp_set_lock_")
      import
      integer(kind=omp_lock_kind), intent(inout) :: lockvar
    end subroutine omp_set_lock

    subroutine omp_unset_lock(lockvar) bind(c, name="omp_unset_lock_")
      import
      integer(kind=omp_lock_kind), intent(inout) :: lockvar
    end subroutine omp_unset_lock

    function omp_test_lock(lockvar) bind(c, name="omp_test_lock_")
      import
      integer(kind=omp_lock_kind), intent(inout) :: lockvar
      logical(kind=omp_logical_kind) :: omp_test_lock
    end function omp_test_lock

    subroutine omp_init_nest_lock(lockvar) bind(c, name="omp_init_nest_lock_")
      import
      integer(kind=omp_nest_lock_kind), intent(out) :: lockvar
    end subroutine omp_init_nest_lock

    subroutine omp_init_nest_lock_with_hint(lockvar, hint) bind(c, name="omp_init_nest_lock_with_hint_")
      import
      integer(kind=omp_nest_lock_kind), intent(out) :: lockvar
      integer(kind=omp_sync_hint_kind), value :: hint
    end subroutine omp_init_nest_lock_with_hint

    subroutine omp_destroy_nest_lock(lockvar) bind(c, name="omp_destroy_nest_lock_")
      import
      integer(kind=omp_nest_lock_kind), intent(inout) :: lockvar
    end subroutine omp_destroy_nest_lock

    subroutine omp_set_nest_lock(lockvar) bind(c, name="omp_set_nest_lock_")
      import
      integer(kind=omp_nest_lock_kind), intent(inout) :: lockvar
    end subroutine omp_set_nest_lock

    subroutine omp_unset_nest_lock(lockvar) bind(c, name="omp_unset_nest_lock_")
      import
      integer(kind=omp_nest_lock_kind), intent(inout) :: lockvar
    end subroutine omp_unset_nest_lock

    function omp_test_nest_lock(lockvar) bind(c, name="omp_test_nest_lock_")
      import
      integer(kind=omp_integer_kind) :: omp_test_nest_lock
      integer(kind=omp_nest_lock_kind), intent(inout) :: lockvar
    end function omp_test_nest_lock

! Timing routines
    function omp_get_wtime() bind(c)
      double precision omp_get_wtime
    end function omp_get_wtime

    function omp_get_wtick() bind(c)
      double precision omp_get_wtick
    end function omp_get_wtick

! Event routine
    subroutine omp_fullfill_event(event) bind(c) ! TODO: is this the correct spelling?
      import
      integer(kind=omp_event_handle_kind) :: event
    end subroutine omp_fullfill_event

! Device Memory Routines

! Memory Management Routines
    function omp_init_allocator(memspace, ntraits, traits)
      import
      integer(kind=omp_memspace_handle_kind), value :: memspace
      integer, value :: ntraits
      type(omp_alloctrait), intent(in) :: traits(*)
      integer(kind=omp_allocator_handle_kind) :: omp_init_allocator
    end function omp_init_allocator

    subroutine omp_destroy_allocator(allocator) bind(c)
      import
      integer(kind=omp_allocator_handle_kind), value :: allocator
    end subroutine omp_destroy_allocator

    subroutine omp_set_default_allocator(allocator) bind(c)
      import
      integer(kind=omp_allocator_handle_kind), value :: allocator
    end subroutine omp_set_default_allocator

    function omp_get_default_allocator()
      import
      integer(kind=omp_allocator_handle_kind) :: omp_get_default_allocator
    end function omp_get_default_allocator

  end interface
