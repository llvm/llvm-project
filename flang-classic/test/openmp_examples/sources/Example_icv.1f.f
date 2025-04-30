! @@name:	icv.1f
! @@type:	F-fixed
! @@compilable:	yes
! @@linkable:	yes
! @@expect:	success
      program icv
      use omp_lib

      call omp_set_nested(.true.)
      call omp_set_max_active_levels(8)
      call omp_set_dynamic(.false.)
      call omp_set_num_threads(2)

!$omp parallel
      call omp_set_num_threads(3)

!$omp parallel
      call omp_set_num_threads(4)
!$omp single
!      The following should print:
!      Inner: max_act_lev= 8 , num_thds= 3 , max_thds= 4
!      Inner: max_act_lev= 8 , num_thds= 3 , max_thds= 4
       print *, "Inner: max_act_lev=", omp_get_max_active_levels(),
     &           ", num_thds=", omp_get_num_threads(),
     &           ", max_thds=", omp_get_max_threads()
!$omp end single
!$omp end parallel

!$omp barrier
!$omp single
!      The following should print:
!      Outer: max_act_lev= 8 , num_thds= 2 , max_thds= 3
       print *, "Outer: max_act_lev=", omp_get_max_active_levels(),
     &           ", num_thds=", omp_get_num_threads(),
     &           ", max_thds=", omp_get_max_threads()
!$omp end single
!$omp end parallel
       end
