! 
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
! 

!          THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT
!   WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT
!   NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR
!   FITNESS FOR A PARTICULAR PURPOSE. 
!
! omp_lib.f90
!
! Generic implementation notes:
! There are a few omp routines which are extended to accept arguments of
! different integer/logical KIND types, such as omp_set_num_threads and
! omp_set_nested.  Routines which have arguments of an OpenMP-defined
! KIND (omp_lock_kind, omp_nest_lock_kind, omp_sched_kind) eannot be
! extended.
! The (f90) method used for implementing the routines as generics is:
! +  present a generic interface 
! +  the specifics are module (contained) procedures
! +  an explicitly named bind C interface is presented as the
!    implementation; the interface represents our C version of the omp
!    routine.
! +  the contained routines, as well as the implementation interfaces,
!    are access private
! +  for windows, the contained routines are forced to be 'cref'
!

#if defined(TARGET_WIN_X8632) || defined(TARGET_WIN_X8664)
#define DECDIR !DEC$
#if defined(TARGET_WIN_X8632)
#define OMP_NAME(A,B) # A
#else
#define OMP_NAME(A,B) # B
#endif
#else
#define DECDIR !!
#endif

!default to 3.1
#ifndef LIBOMP_YEAR_MONTH
#define LIBOMP_YEAR_MONTH 201107
#endif

module omp_lib_kinds
  integer, parameter :: omp_integer_kind = 4
  integer, parameter :: omp_logical_kind = 4
#if defined(TARGET_KMPC)
  integer, parameter :: omp_lock_kind = 8
#else
  integer, parameter :: omp_lock_kind = 4
#endif
  integer, parameter :: omp_nest_lock_kind = 8
  integer, parameter :: omp_sched_kind  = 4
  integer, parameter :: omp_real_kind          = 4
  integer, parameter :: omp_proc_bind_kind     = omp_integer_kind
  integer, parameter :: omp_lock_hint_kind     = omp_integer_kind

  integer(kind=omp_sched_kind), parameter :: omp_sched_static  = 1
  integer(kind=omp_sched_kind), parameter :: omp_sched_dynamic = 2
  integer(kind=omp_sched_kind), parameter :: omp_sched_guided  = 3
  integer(kind=omp_sched_kind), parameter :: omp_sched_auto    = 4

  integer (kind=omp_proc_bind_kind), parameter :: omp_proc_bind_false = 0
  integer (kind=omp_proc_bind_kind), parameter :: omp_proc_bind_true = 1
  integer (kind=omp_proc_bind_kind), parameter :: omp_proc_bind_master = 2
  integer (kind=omp_proc_bind_kind), parameter :: omp_proc_bind_close = 3
  integer (kind=omp_proc_bind_kind), parameter :: omp_proc_bind_spread = 4

  integer (kind=omp_lock_hint_kind), parameter :: omp_lock_hint_none           = 0
  integer (kind=omp_lock_hint_kind), parameter :: omp_lock_hint_uncontended    = 1
  integer (kind=omp_lock_hint_kind), parameter :: omp_lock_hint_contended      = 2
  integer (kind=omp_lock_hint_kind), parameter :: omp_lock_hint_nonspeculative = 4
  integer (kind=omp_lock_hint_kind), parameter :: omp_lock_hint_speculative    = 8


end module omp_lib_kinds

module omp_lib
  use omp_lib_kinds
#ifdef PGDLL
!!! Now that there's code in the module, cannot use the dllexport trick
!DEC$ ATTRIBUTES DLLEXPORT :: omp_lib
#endif

! versoin 3.1 201107
! version 4.0 201307
! version 4.5 201511

! right now make llvm x86 same as native
#if defined(TARGET_KMPC)
  integer, parameter :: openmp_version = 201307
#else
  integer, parameter :: openmp_version = 201307
#endif
  interface
    subroutine omp_destroy_lock ( svar )
    use omp_lib_kinds
    integer ( kind=omp_lock_kind ), intent(inout) :: svar
DECDIR ATTRIBUTES C :: omp_destroy_lock
DECDIR ATTRIBUTES REFERENCE :: omp_destroy_lock
DECDIR ATTRIBUTES ALIAS:OMP_NAME(_omp_destroy_lock_,omp_destroy_lock_) :: omp_destroy_lock
    end subroutine omp_destroy_lock
  end interface

  interface
    subroutine omp_destroy_nest_lock ( nvar )
    use omp_lib_kinds
    integer ( kind=omp_nest_lock_kind ), intent(inout) :: nvar
DECDIR ATTRIBUTES C :: omp_destroy_nest_lock
DECDIR ATTRIBUTES REFERENCE :: omp_destroy_nest_lock
DECDIR ATTRIBUTES ALIAS:OMP_NAME(_omp_destroy_nest_lock_,omp_destroy_nest_lock_) :: omp_destroy_nest_lock
    end subroutine omp_destroy_nest_lock
  end interface

  interface
    function omp_get_dynamic ()
    use omp_lib_kinds
    logical ( kind=omp_logical_kind ) :: omp_get_dynamic
DECDIR ATTRIBUTES C :: omp_get_dynamic
DECDIR ATTRIBUTES REFERENCE :: omp_get_dynamic
DECDIR ATTRIBUTES ALIAS:OMP_NAME(_omp_get_dynamic_,omp_get_dynamic_) :: omp_get_dynamic
    end function omp_get_dynamic
  end interface

  interface
    function omp_get_max_threads ()
    use omp_lib_kinds
    integer ( kind=omp_integer_kind ) :: omp_get_max_threads
DECDIR ATTRIBUTES C :: omp_get_max_threads
DECDIR ATTRIBUTES REFERENCE :: omp_get_max_threads
DECDIR ATTRIBUTES ALIAS:OMP_NAME(_omp_get_max_threads_,omp_get_max_threads_) :: omp_get_max_threads
    end function omp_get_max_threads
  end interface

  interface
    function omp_get_nested ()
    use omp_lib_kinds
    logical ( kind=omp_logical_kind ) :: omp_get_nested
DECDIR ATTRIBUTES C :: omp_get_nested
DECDIR ATTRIBUTES REFERENCE :: omp_get_nested
DECDIR ATTRIBUTES ALIAS:OMP_NAME(_omp_get_nested_,omp_get_nested_) :: omp_get_nested
    end function omp_get_nested
  end interface

  interface omp_set_schedule
    module procedure oss1, oss2, oss4, oss8
  end interface
  interface
    subroutine ompsetschedule(kind, modifier) bind(C, name='omp_set_schedule')
      use  omp_lib_kinds
      integer(omp_sched_kind), value :: kind
      integer(4)             , value :: modifier
    end subroutine
  end interface
  private oss1, oss2, oss4, oss8, ompsetschedule

  interface omp_get_schedule
    module procedure ogs1, ogs2, ogs4, ogs8
  end interface
  interface
    subroutine ompgetschedule(kind, modifier) bind(C, name='omp_get_schedule')
      use  omp_lib_kinds
      integer(omp_sched_kind) :: kind
      integer(4)              :: modifier
    end subroutine
  end interface
  private ogs1, ogs2, ogs4, ogs8, ompgetschedule

  interface
    function omp_get_thread_limit ()
    use omp_lib_kinds
    integer ( kind=omp_integer_kind ) :: omp_get_thread_limit
DECDIR ATTRIBUTES C :: omp_get_thread_limit
DECDIR ATTRIBUTES REFERENCE :: omp_get_thread_limit
DECDIR ATTRIBUTES ALIAS:OMP_NAME(_omp_get_thread_limit_,omp_get_thread_limit_) :: omp_get_thread_limit
    end function omp_get_thread_limit
  end interface

  interface
    function omp_get_num_procs ()
    use omp_lib_kinds
    integer ( kind=omp_integer_kind ) :: omp_get_num_procs
DECDIR ATTRIBUTES C :: omp_get_num_procs
DECDIR ATTRIBUTES REFERENCE :: omp_get_num_procs
DECDIR ATTRIBUTES ALIAS:OMP_NAME(_omp_get_num_procs_,omp_get_num_procs_) :: omp_get_num_procs
    end function omp_get_num_procs
  end interface

  interface
    function omp_get_num_threads ()
    use omp_lib_kinds
    integer ( kind=omp_integer_kind ) :: omp_get_num_threads
DECDIR ATTRIBUTES C :: omp_get_num_threads
DECDIR ATTRIBUTES REFERENCE :: omp_get_num_threads
DECDIR ATTRIBUTES ALIAS:OMP_NAME(_omp_get_num_threads_,omp_get_num_threads_) :: omp_get_num_threads
    end function omp_get_num_threads
  end interface

  interface
    function omp_get_thread_num ()
    use omp_lib_kinds
    integer ( kind=omp_integer_kind ) :: omp_get_thread_num
DECDIR ATTRIBUTES C :: omp_get_thread_num
DECDIR ATTRIBUTES REFERENCE :: omp_get_thread_num
DECDIR ATTRIBUTES ALIAS:OMP_NAME(_omp_get_thread_num_,omp_get_thread_num_) :: omp_get_thread_num
    end function omp_get_thread_num
  end interface

  interface
    function omp_get_wtick ()
    double precision :: omp_get_wtick
DECDIR ATTRIBUTES C :: omp_get_wtick
DECDIR ATTRIBUTES REFERENCE :: omp_get_wtick
DECDIR ATTRIBUTES ALIAS:OMP_NAME(_omp_get_wtick_,omp_get_wtick_) :: omp_get_wtick
    end function omp_get_wtick
  end interface

  interface
    function omp_get_wtime ()
    double precision :: omp_get_wtime
DECDIR ATTRIBUTES C :: omp_get_wtime
DECDIR ATTRIBUTES REFERENCE :: omp_get_wtime
DECDIR ATTRIBUTES ALIAS:OMP_NAME(_omp_get_wtime_,omp_get_wtime_) :: omp_get_wtime
    end function omp_get_wtime
  end interface

  interface
    subroutine omp_init_lock ( svar )
    use omp_lib_kinds
    integer ( kind=omp_lock_kind ), intent(out) :: svar
DECDIR ATTRIBUTES C :: omp_init_lock
DECDIR ATTRIBUTES REFERENCE :: omp_init_lock
DECDIR ATTRIBUTES ALIAS:OMP_NAME(_omp_init_lock_,omp_init_lock_) :: omp_init_lock
    end subroutine omp_init_lock
  end interface

  interface
    subroutine omp_init_nest_lock ( nvar )
    use omp_lib_kinds
    integer ( kind=omp_nest_lock_kind ), intent(out) :: nvar
DECDIR ATTRIBUTES C :: omp_init_nest_lock
DECDIR ATTRIBUTES REFERENCE :: omp_init_nest_lock
DECDIR ATTRIBUTES ALIAS:OMP_NAME(_omp_init_nest_lock_,omp_init_nest_lock_) :: omp_init_nest_lock
    end subroutine omp_init_nest_lock
  end interface

  interface
    function omp_in_parallel ()
    use omp_lib_kinds
    logical ( kind=omp_logical_kind ) :: omp_in_parallel
DECDIR ATTRIBUTES C :: omp_in_parallel
DECDIR ATTRIBUTES REFERENCE :: omp_in_parallel
DECDIR ATTRIBUTES ALIAS:OMP_NAME(_omp_in_parallel_,omp_in_parallel_) :: omp_in_parallel
    end function omp_in_parallel
  end interface

  interface
    function omp_in_final ()
    use omp_lib_kinds
    logical ( kind=omp_logical_kind ) :: omp_in_final
DECDIR ATTRIBUTES C :: omp_in_final
DECDIR ATTRIBUTES REFERENCE :: omp_in_final
DECDIR ATTRIBUTES ALIAS:OMP_NAME(_omp_in_final_,omp_in_final_) :: omp_in_final
    end function omp_in_final
  end interface

  interface omp_set_dynamic
    module procedure osd1, osd2, osd4, osd8
  end interface
  interface
    subroutine ompsetdynamic(dynamic_threads) bind(C, name='omp_set_dynamic')
      logical(4), value :: dynamic_threads
    end subroutine
  end interface
  private osd1, osd2, osd4, osd8, ompsetdynamic

  interface
    subroutine omp_set_lock ( svar )
    use omp_lib_kinds
    integer ( kind=omp_lock_kind ), intent(inout) :: svar
DECDIR ATTRIBUTES C :: omp_set_lock
DECDIR ATTRIBUTES REFERENCE :: omp_set_lock
DECDIR ATTRIBUTES ALIAS:OMP_NAME(_omp_set_lock_,omp_set_lock_) :: omp_set_lock
    end subroutine omp_set_lock
  end interface

  interface
    subroutine omp_set_nest_lock ( nvar )
    use omp_lib_kinds
    integer ( kind=omp_nest_lock_kind ), intent(inout) :: nvar
DECDIR ATTRIBUTES C :: omp_set_nest_lock
DECDIR ATTRIBUTES REFERENCE :: omp_set_nest_lock
DECDIR ATTRIBUTES ALIAS:OMP_NAME(_omp_set_nest_lock_,omp_set_nest_lock_) :: omp_set_nest_lock
    end subroutine omp_set_nest_lock
  end interface

  interface
    function omp_test_lock ( svar )
    use omp_lib_kinds
    logical ( kind=omp_logical_kind ) :: omp_test_lock
    integer ( kind=omp_lock_kind ), intent(inout) :: svar
DECDIR ATTRIBUTES C :: omp_test_lock
DECDIR ATTRIBUTES REFERENCE :: omp_test_lock
DECDIR ATTRIBUTES ALIAS:OMP_NAME(_omp_test_lock_,omp_test_lock_) :: omp_test_lock
    end function omp_test_lock
  end interface

  interface
    function omp_test_nest_lock ( nvar )
    use omp_lib_kinds
    integer ( kind=omp_integer_kind ) :: omp_test_nest_lock
    integer ( kind=omp_nest_lock_kind ), intent(inout) :: nvar
DECDIR ATTRIBUTES C :: omp_test_nest_lock
DECDIR ATTRIBUTES REFERENCE :: omp_test_nest_lock
DECDIR ATTRIBUTES ALIAS:OMP_NAME(_omp_test_nest_lock_,omp_test_nest_lock_) :: omp_test_nest_lock
    end function omp_test_nest_lock
  end interface

  interface
    subroutine omp_unset_lock ( svar )
    use omp_lib_kinds
    integer ( kind=omp_lock_kind ), intent(inout) :: svar
DECDIR ATTRIBUTES C :: omp_unset_lock
DECDIR ATTRIBUTES REFERENCE :: omp_unset_lock
DECDIR ATTRIBUTES ALIAS:OMP_NAME(_omp_unset_lock_,omp_unset_lock_) :: omp_unset_lock
    end subroutine omp_unset_lock
  end interface

  interface
    subroutine omp_unset_nest_lock ( nvar )
    use omp_lib_kinds
    integer ( kind=omp_nest_lock_kind ), intent(inout) :: nvar
DECDIR ATTRIBUTES C :: omp_unset_nest_lock
DECDIR ATTRIBUTES REFERENCE :: omp_unset_nest_lock
DECDIR ATTRIBUTES ALIAS:OMP_NAME(_omp_unset_nest_lock_,omp_unset_nest_lock_) :: omp_get_max_threads
    end subroutine omp_unset_nest_lock
  end interface

  interface omp_set_nested
    module procedure osn1, osn2, osn4, osn8
  end interface
  interface
    subroutine ompsetnested(s) bind(C, name='omp_set_nested')
      logical(4), value :: s
    end subroutine
  end interface
  private osn1, osn2, osn4, osn8, ompsetnested

  interface omp_set_num_threads
    module procedure osnt1, osnt2, osnt4, osnt8
  end interface
  interface
    subroutine ompsetnumthreads(s) bind(C, name='omp_set_num_threads')
      integer(4), value :: s
    end subroutine
  end interface
  private osnt1, osnt2, osnt4, osnt8, ompsetnumthreads

  interface
    function omp_get_max_active_levels() 
    use omp_lib_kinds 
    integer (kind=omp_integer_kind) :: omp_get_max_active_levels 
DECDIR ATTRIBUTES C :: omp_get_max_active_levels
DECDIR ATTRIBUTES REFERENCE :: omp_get_max_active_levels
DECDIR ATTRIBUTES ALIAS:OMP_NAME(_omp_get_max_active_levels_,omp_get_max_active_levels_) :: omp_get_max_active_levels
    end function omp_get_max_active_levels 
  end interface

  interface omp_set_max_active_levels
    module procedure osmal1, osmal2, osmal4, osmal8
  end interface
  interface
    subroutine ompsetmaxactivelevels(max_levels) bind(C, name='omp_set_max_active_levels')
      integer(4), value :: max_levels
    end subroutine
  end interface
  private osmal1, osmal2, osmal4, osmal8, ompsetmaxactivelevels

  interface
    function omp_get_level() 
    use omp_lib_kinds 
    integer (kind=omp_integer_kind) :: omp_get_level 
DECDIR ATTRIBUTES C :: omp_get_level
DECDIR ATTRIBUTES REFERENCE :: omp_get_level
DECDIR ATTRIBUTES ALIAS:OMP_NAME(_omp_get_level_,omp_get_level_) :: omp_get_level
    end function omp_get_level 
  end  interface

  interface omp_get_ancestor_thread_num
    module procedure ogatn1, ogatn2, ogatn4, ogatn8
  end interface
  interface
    function ompgetancestorthreadnum(level) bind(C, name='omp_get_ancestor_thread_num')
      use omp_lib_kinds 
      integer(4), value :: level 
      integer(kind=omp_integer_kind) :: ompgetancestorthreadnum 
    end function
  end interface
  private ogatn1, ogatn2, ogatn4, ogatn8, ompgetancestorthreadnum

  interface omp_get_team_size
    module procedure ogts1, ogts2, ogts4, ogts8
  end interface
  interface
    function ompgetteamsize(level) bind(C, name='omp_get_team_size')
      use omp_lib_kinds 
      integer(4), value :: level 
      integer(kind=omp_integer_kind) :: ompgetteamsize 
    end function
  end interface
  private ogts1, ogts2, ogts4, ogts8, ompgetteamsize

  interface
    function omp_get_active_level() 
    use omp_lib_kinds 
    integer (kind=omp_integer_kind) :: omp_get_active_level 
DECDIR ATTRIBUTES C :: omp_get_active_level
DECDIR ATTRIBUTES REFERENCE :: omp_get_active_level
DECDIR ATTRIBUTES ALIAS:OMP_NAME(_omp_get_active_level_,omp_get_active_level_) :: omp_get_active_level
    end function omp_get_active_level 
  end interface

!4.X

  interface

    function omp_get_proc_bind() bind(C, name='omp_get_proc_bind')
      use omp_lib_kinds
      integer (kind=omp_proc_bind_kind) omp_get_proc_bind
    end function omp_get_proc_bind

    function omp_get_num_places() bind(C, name='omp_get_num_places')
      use omp_lib_kinds
      integer (kind=omp_integer_kind) omp_get_num_places
    end function omp_get_num_places

    function omp_get_place_num_procs(place_num) bind(C, name='omp_get_place_num_procs')
      use omp_lib_kinds
      integer (kind=omp_integer_kind), value :: place_num
      integer (kind=omp_integer_kind) omp_get_place_num_procs
    end function omp_get_place_num_procs

    subroutine omp_get_place_proc_ids(place_num, ids) bind(C, name='omp_get_place_proc_ids')
      use omp_lib_kinds
      integer (kind=omp_integer_kind), value :: place_num
      integer ids(*)
    end subroutine omp_get_place_proc_ids

    function omp_get_place_num() bind(C, name='omp_get_place_num')
      use omp_lib_kinds
      integer (kind=omp_integer_kind) omp_get_place_num
    end function omp_get_place_num

    function omp_get_partition_num_places() bind(C, name='omp_get_partition_num_places')
      use omp_lib_kinds
      integer (kind=omp_integer_kind) omp_get_partition_num_places
    end function omp_get_partition_num_places

    subroutine omp_get_partition_place_nums(place_nums) bind(C, name='omp_get_partition_place_nums')
      use omp_lib_kinds
      integer place_nums(*)
    end subroutine omp_get_partition_place_nums

    function omp_get_default_device() bind(C, name='omp_get_default_device')
      use omp_lib_kinds
      integer (kind=omp_integer_kind) omp_get_default_device
    end function omp_get_default_device

    subroutine omp_set_default_device(device_num) bind(C, name='omp_set_default_device')
      use omp_lib_kinds
      integer (kind=omp_integer_kind), value :: device_num
    end subroutine omp_set_default_device

    function omp_get_num_devices() bind(C, name='omp_get_num_devices')
      use omp_lib_kinds
      integer (kind=omp_integer_kind) omp_get_num_devices
    end function omp_get_num_devices

    function omp_get_num_teams() bind(C, name='omp_get_num_teams')
      use omp_lib_kinds
      integer (kind=omp_integer_kind) omp_get_num_teams
    end function omp_get_num_teams

    function omp_get_team_num() bind(C, name='omp_get_team_num')
      use omp_lib_kinds
      integer (kind=omp_integer_kind) omp_get_team_num
    end function omp_get_team_num

    function omp_get_cancellation() bind(C, name='omp_get_cancellation')
      use omp_lib_kinds
      integer (kind=omp_integer_kind) omp_get_cancellation
    end function omp_get_cancellation

    function omp_is_initial_device() bind(C, name='omp_is_initial_device')
      use omp_lib_kinds
      logical (kind=omp_logical_kind) omp_is_initial_device
    end function omp_is_initial_device

    function omp_get_initial_device() bind(C, name='omp_get_initial_device')
      use omp_lib_kinds
      integer (kind=omp_integer_kind) omp_get_initial_device
    end function omp_get_initial_device

    function omp_get_max_task_priority() bind(C, name='omp_get_max_task_priority')
      use omp_lib_kinds
      integer (kind=omp_integer_kind) omp_get_max_task_priority
    end function omp_get_max_task_priority

    subroutine omp_init_lock_with_hint(svar, hint) bind(C, name='omp_init_lock_with_hint')
      use omp_lib_kinds
      integer (kind=omp_lock_kind) svar
      integer (kind=omp_lock_hint_kind), value :: hint
    end subroutine omp_init_lock_with_hint

    subroutine omp_init_nest_lock_with_hint(nvar, hint) bind(C, name='omp_init_nest_lock_with_hint')
      use omp_lib_kinds
      integer (kind=omp_nest_lock_kind) nvar
      integer (kind=omp_lock_hint_kind), value :: hint
    end subroutine omp_init_nest_lock_with_hint

  end interface



!!!  integer :: omp_lib_export
!!!!dec$ attributes dllexport :: omp_lib_export

contains

! ------  generic omp_set_schedule
!         oss1, oss2, oss4, oss8
  subroutine oss1( kind, modifier )
    use omp_lib_kinds
    integer(kind=omp_sched_kind), intent(in) :: kind
    integer(kind=1), intent(in)              :: modifier
DECDIR ATTRIBUTES C :: oss1
DECDIR ATTRIBUTES REFERENCE :: oss1
    call ompsetschedule(kind, int(modifier,4))
  end subroutine oss1
  subroutine oss2( kind, modifier )
    use omp_lib_kinds
    integer(kind=omp_sched_kind), intent(in) :: kind
    integer(kind=2), intent(in)              :: modifier
DECDIR ATTRIBUTES C :: oss2
DECDIR ATTRIBUTES REFERENCE :: oss2
    call ompsetschedule(kind, int(modifier,4))
  end subroutine oss2
  subroutine oss4( kind, modifier )
    use omp_lib_kinds
    integer(kind=omp_sched_kind), intent(in) :: kind
    integer(kind=4), intent(in)              :: modifier
DECDIR ATTRIBUTES C :: oss4
DECDIR ATTRIBUTES REFERENCE :: oss4
    call ompsetschedule(kind, int(modifier,4))
  end subroutine oss4
  subroutine oss8( kind, modifier )
    use omp_lib_kinds
    integer(kind=omp_sched_kind), intent(in) :: kind
    integer(kind=8), intent(in)              :: modifier
DECDIR ATTRIBUTES C :: oss8
DECDIR ATTRIBUTES REFERENCE :: oss8
    call ompsetschedule(kind, int(modifier,4))
  end subroutine oss8
! ------  endgeneric omp_set_schedule

! ------  generic omp_get_schedule
!         ogs1, ogs2, ogs4, ogs8
  subroutine ogs1( kind, modifier )
    use omp_lib_kinds
    integer(kind=omp_sched_kind) :: kind
    integer(kind=1)              :: modifier
    integer(4) :: temp
DECDIR ATTRIBUTES C :: ogs1
DECDIR ATTRIBUTES REFERENCE :: ogs1
    call ompgetschedule(kind, temp)
    modifier = temp
  end subroutine ogs1
  subroutine ogs2( kind, modifier )
    use omp_lib_kinds
    integer(kind=omp_sched_kind) :: kind
    integer(kind=2)              :: modifier
    integer(4) :: temp
DECDIR ATTRIBUTES C :: ogs2
DECDIR ATTRIBUTES REFERENCE :: ogs2
    call ompgetschedule(kind, temp)
    modifier = temp
  end subroutine ogs2
  subroutine ogs4( kind, modifier )
    use omp_lib_kinds
    integer(kind=omp_sched_kind) :: kind
    integer(kind=4)              :: modifier
    integer(4) :: temp
DECDIR ATTRIBUTES C :: ogs4
DECDIR ATTRIBUTES REFERENCE :: ogs4
    call ompgetschedule(kind, temp)
    modifier = temp
  end subroutine ogs4
  subroutine ogs8( kind, modifier )
    use omp_lib_kinds
    integer(kind=omp_sched_kind) :: kind
    integer(kind=8)              :: modifier
    integer(4) :: temp
DECDIR ATTRIBUTES C :: ogs8
DECDIR ATTRIBUTES REFERENCE :: ogs8
    call ompgetschedule(kind, temp)
    modifier = temp
  end subroutine ogs8
! ------  endgeneric omp_set_schedule

! ------  generic omp_set_dynamic
  subroutine osd1( dynamic_threads )
    logical(kind=1), intent(in) :: dynamic_threads
DECDIR ATTRIBUTES C :: osd1
DECDIR ATTRIBUTES REFERENCE :: osd1
    call ompsetdynamic(logical(dynamic_threads,4))
  end subroutine osd1
  subroutine osd2( dynamic_threads )
    logical(kind=2), intent(in) :: dynamic_threads
DECDIR ATTRIBUTES C :: osd2
DECDIR ATTRIBUTES REFERENCE :: osd2
    call ompsetdynamic(logical(dynamic_threads,4))
  end subroutine osd2
  subroutine osd4( dynamic_threads )
    logical(kind=4), intent(in) :: dynamic_threads
DECDIR ATTRIBUTES C :: osd4
DECDIR ATTRIBUTES REFERENCE :: osd4
    call ompsetdynamic(logical(dynamic_threads,4))
  end subroutine osd4
  subroutine osd8( dynamic_threads )
    logical(kind=8), intent(in) :: dynamic_threads
DECDIR ATTRIBUTES C :: osd8
DECDIR ATTRIBUTES REFERENCE :: osd8
    call ompsetdynamic(logical(dynamic_threads,4))
  end subroutine osd8
! ------  endgeneric omp_set_dynamic

! ------  generic omp_set_nested
!         osn1, osn2, osn4, osn8
  subroutine osn1( nested )
    logical(kind=1), intent(in) :: nested
DECDIR ATTRIBUTES C :: osn1
DECDIR ATTRIBUTES REFERENCE :: osn1
    call ompsetnested(logical(nested,4))
  end subroutine osn1
  subroutine osn2( nested )
    logical(kind=2), intent(in) :: nested
DECDIR ATTRIBUTES C :: osn2
DECDIR ATTRIBUTES REFERENCE :: osn2
    call ompsetnested(logical(nested,4))
  end subroutine osn2
  subroutine osn4( nested )
    logical(kind=4), intent(in) :: nested
DECDIR ATTRIBUTES C :: osn4
DECDIR ATTRIBUTES REFERENCE :: osn4
    call ompsetnested(logical(nested,4))
  end subroutine osn4
  subroutine osn8( nested )
    logical(kind=8), intent(in) :: nested
DECDIR ATTRIBUTES C :: osn8
DECDIR ATTRIBUTES REFERENCE :: osn8
    call ompsetnested(logical(nested,4))
  end subroutine osn8
! ------  endgeneric omp_set_nested

! ------  generic omp_set_num_threads
!         osnt1, osnt2, osnt4, osnt8
  subroutine osnt1( num_threads )
    integer(kind=1), intent(in) :: num_threads
DECDIR ATTRIBUTES C :: osnt1
DECDIR ATTRIBUTES REFERENCE :: osnt1
    call ompsetnumthreads(int(num_threads,4))
  end subroutine osnt1
  subroutine osnt2( num_threads )
    integer(kind=2), intent(in) :: num_threads
DECDIR ATTRIBUTES C :: osnt2
DECDIR ATTRIBUTES REFERENCE :: osnt2
    call ompsetnumthreads(int(num_threads,4))
  end subroutine osnt2
  subroutine osnt4( num_threads )
    integer(kind=4), intent(in) :: num_threads
DECDIR ATTRIBUTES C :: osnt4
DECDIR ATTRIBUTES REFERENCE :: osnt4
    call ompsetnumthreads(int(num_threads,4))
  end subroutine osnt4
  subroutine osnt8( num_threads )
    integer(kind=8), intent(in) :: num_threads
DECDIR ATTRIBUTES C :: osnt8
DECDIR ATTRIBUTES REFERENCE :: osnt8
    call ompsetnumthreads(int(num_threads,4))
  end subroutine osnt8
! ------  endgeneric omp_set_num_threads

! ------  generic omp_set_max_active_levels
!         osmal1, osmal2, osmal4, osmal8
  subroutine osmal1( max_levels )
    integer(kind=1), intent(in) :: max_levels
DECDIR ATTRIBUTES C :: osmal1
DECDIR ATTRIBUTES REFERENCE :: osmal1
    call ompsetmaxactivelevels(int(max_levels,4))
  end subroutine osmal1
  subroutine osmal2( max_levels )
    integer(kind=2), intent(in) :: max_levels
DECDIR ATTRIBUTES C :: osmal2
DECDIR ATTRIBUTES REFERENCE :: osmal2
    call ompsetmaxactivelevels(int(max_levels,4))
  end subroutine osmal2
  subroutine osmal4( max_levels )
    integer(kind=4), intent(in) :: max_levels
DECDIR ATTRIBUTES C :: osmal4
DECDIR ATTRIBUTES REFERENCE :: osmal4
    call ompsetmaxactivelevels(int(max_levels,4))
  end subroutine osmal4
  subroutine osmal8( max_levels )
    integer(kind=8), intent(in) :: max_levels
DECDIR ATTRIBUTES C :: osmal8
DECDIR ATTRIBUTES REFERENCE :: osmal8
    call ompsetmaxactivelevels(int(max_levels,4))
  end subroutine osmal8
! ------  endgeneric omp_set_max_active_levels

! ------  generic omp_get_ancestor_thread_num
!         ogatn1, ogatn2, ogatn4, ogatn8
  function ogatn1( level )
    use omp_lib_kinds
    integer(kind=1), intent(in) :: level
    integer(kind=omp_integer_kind) :: ogatn1
DECDIR ATTRIBUTES C :: ogatn1
DECDIR ATTRIBUTES REFERENCE :: ogatn1
    ogatn1 = ompgetancestorthreadnum(int(level,4))
  end function ogatn1
  function ogatn2( level )
    use omp_lib_kinds
    integer(kind=2), intent(in) :: level
    integer(kind=omp_integer_kind) :: ogatn2
DECDIR ATTRIBUTES C :: ogatn2
DECDIR ATTRIBUTES REFERENCE :: ogatn2
    ogatn2 = ompgetancestorthreadnum(int(level,4))
  end function ogatn2
  function ogatn4( level )
    use omp_lib_kinds
    integer(kind=4), intent(in) :: level
    integer(kind=omp_integer_kind) :: ogatn4
DECDIR ATTRIBUTES C :: ogatn4
DECDIR ATTRIBUTES REFERENCE :: ogatn4
    ogatn4 = ompgetancestorthreadnum(int(level,4))
  end function ogatn4
  function ogatn8( level )
    use omp_lib_kinds
    integer(kind=8), intent(in) :: level
    integer(kind=omp_integer_kind) :: ogatn4
DECDIR ATTRIBUTES C :: ogatn8
DECDIR ATTRIBUTES REFERENCE :: ogatn8
    ogatn8 = ompgetancestorthreadnum(int(level,4))
  end function ogatn8
! ------  endgeneric omp_get_ancestor_thread_num

! ------  generic omp_get_team_size
!         ogts1, ogts2, ogts4, ogts8
  function ogts1( level )
    use omp_lib_kinds
    integer(kind=1), intent(in) :: level
    integer(kind=omp_integer_kind) :: ogts1
DECDIR ATTRIBUTES C :: ogts1
DECDIR ATTRIBUTES REFERENCE :: ogts1
    ogts1 = ompgetteamsize(int(level,4))
  end function ogts1
  function ogts2( level )
    use omp_lib_kinds
    integer(kind=2), intent(in) :: level
    integer(kind=omp_integer_kind) :: ogts2
DECDIR ATTRIBUTES C :: ogts2
DECDIR ATTRIBUTES REFERENCE :: ogts2
    ogts2 = ompgetteamsize(int(level,4))
  end function ogts2
  function ogts4( level )
    use omp_lib_kinds
    integer(kind=4), intent(in) :: level
    integer(kind=omp_integer_kind) :: ogts4
DECDIR ATTRIBUTES C :: ogts4
DECDIR ATTRIBUTES REFERENCE :: ogts4
    ogts4 = ompgetteamsize(int(level,4))
  end function ogts4
  function ogts8( level )
    use omp_lib_kinds
    integer(kind=8), intent(in) :: level
    integer(kind=omp_integer_kind) :: ogts4
DECDIR ATTRIBUTES C :: ogts8
DECDIR ATTRIBUTES REFERENCE :: ogts8
    ogts8 = ompgetteamsize(int(level,4))
  end function ogts8
! ------  endgeneric omp_get_team_size

end module omp_lib
