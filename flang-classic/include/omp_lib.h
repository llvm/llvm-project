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
! omp_lib.h
!

       integer omp_lock_kind
       parameter ( omp_lock_kind = 4 )
       integer omp_nest_lock_kind
       parameter ( omp_nest_lock_kind = 8 )

       integer omp_integer_kind
       parameter ( omp_integer_kind = 4 )
       integer omp_logical_kind
       parameter ( omp_logical_kind = 4 )

       integer omp_sched_kind 
       parameter ( omp_sched_kind = 4) 
       integer ( omp_sched_kind ) omp_sched_static 
       parameter ( omp_sched_static = 1 ) 
       integer ( omp_sched_kind ) omp_sched_dynamic 
       parameter ( omp_sched_dynamic = 2 ) 
       integer ( omp_sched_kind ) omp_sched_guided 
       parameter ( omp_sched_guided = 3 ) 
       integer ( omp_sched_kind ) omp_sched_auto 
       parameter ( omp_sched_auto = 4 ) 

       integer omp_proc_bind_kind 
       parameter ( omp_proc_bind_kind = omp_integer_kind )
       integer (kind=omp_proc_bind_kind) omp_proc_bind_false
       parameter ( omp_proc_bind_false = 0 )
       integer (kind=omp_proc_bind_kind) omp_proc_bind_true
       parameter ( omp_proc_bind_true = 1 )
       integer (kind=omp_proc_bind_kind) omp_proc_bind_master
       parameter ( omp_proc_bind_master = 2 )
       integer (kind=omp_proc_bind_kind) omp_proc_bind_close
       parameter ( omp_proc_bind_close = 3 )
       integer (kind=omp_proc_bind_kind) omp_proc_bind_spread 
       parameter ( omp_proc_bind_spread = 4 )

       integer omp_lock_hint_kind
       parameter ( omp_lock_hint_kind = omp_integer_kind )
       integer (kind=omp_lock_hint_kind) omp_lock_hint_none
       parameter ( omp_lock_hint_none = 0 )
       integer (kind=omp_lock_hint_kind) omp_lock_hint_uncontended
       parameter ( omp_lock_hint_uncontended = 1 )
       integer (kind=omp_lock_hint_kind) omp_lock_hint_contended 
       parameter ( omp_lock_hint_contended = 2 )
       integer (kind=omp_lock_hint_kind) omp_lock_hint_nonspeculative
       parameter ( omp_lock_hint_nonspeculative = 4 )
       integer (kind=omp_lock_hint_kind) omp_lock_hint_speculative 
       parameter ( omp_lock_hint_speculative = 8 )

! version > 3.0 201101
! version > 4.0 201307
! else          200505

       integer openmp_version
       parameter ( openmp_version = 201307 )
       external omp_destroy_lock
       external omp_destroy_nest_lock

       external omp_get_dynamic
       logical( omp_logical_kind ) omp_get_dynamic

       external omp_get_max_threads
       integer( omp_logical_kind ) omp_get_max_threads

       external omp_get_nested
       logical( omp_logical_kind ) omp_get_nested

       external omp_get_num_procs
       integer( omp_integer_kind ) omp_get_num_procs

       external omp_get_num_threads
       integer( omp_integer_kind ) omp_get_num_threads

       external omp_get_thread_num
       integer( omp_integer_kind ) omp_get_thread_num

       external omp_get_wtick
       double precision omp_get_wtick

       external omp_get_wtime
       double precision omp_get_wtime
       external omp_init_lock
       external omp_init_nest_lock

       external omp_in_parallel
       logical( omp_logical_kind ) omp_in_parallel

       external omp_in_final
       logical( omp_logical_kind ) omp_in_final

       external omp_set_dynamic
       external omp_set_lock
       external omp_set_nest_lock
       external omp_set_nested
       external omp_set_num_threads

       external omp_test_lock
       logical( omp_logical_kind ) omp_test_lock

       external omp_test_nest_lock
       integer( omp_integer_kind ) omp_test_nest_lock

       external omp_unset_lock
       external omp_unset_nest_lock

	external omp_set_dynamic 
!	external omp_get_dynamic 
!	logical omp_get_dynamic 
	external omp_set_nested 
!	external omp_get_nested 
!	logical omp_get_nested 
	external omp_set_schedule 
	external omp_get_schedule 
	external omp_get_thread_limit 
	integer omp_get_thread_limit 
	external omp_set_max_active_levels 
	external omp_get_max_active_levels 
	integer omp_get_max_active_levels 
	external omp_get_level 
	integer omp_get_level 
	external omp_get_ancestor_thread_num 
	integer omp_get_ancestor_thread_num 
	external omp_get_team_size 
	integer omp_get_team_size 
	external omp_get_active_level 
	integer omp_get_active_level 

!4.X
	external omp_get_cancellation
	logical omp_get_cancellation

	external omp_get_proc_bind
	integer(omp_proc_bind_kind) omp_get_proc_bind

	external omp_get_num_places
	integer omp_get_num_places

	external omp_get_place_num_procs
	integer omp_get_place_num_procs

	external omp_get_place_proc_ids

	external omp_get_place_num
	integer omp_get_place_num

	external omp_get_partition_num_places
	integer omp_gte_partition_num_places

	external omp_get_partition_place_nums
	
	external omp_set_default_device

	external omp_get_default_device
	integer omp_get_default_device

	external omp_get_num_devices
	integer omp_get_num_devices

	external omp_get_num_teams
	integer omp_get_num_teams

	external omp_get_team_num
	integer omp_get_team_num

	external omp_is_initial_device
	integer omp_is_initial_device

	external omp_get_get_initial_device
	integer omp_get_initial_device

	external omp_get_max_task_priority
	integer omp_get_max_task_priority

	external omp_init_lock_with_hint
	external omp_init_nest_lock_with_hint

