! @@name:	tasking.8f
! @@type:	F-fixed
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
      module example
      integer tp
!$omp threadprivate(tp)
      integer var
      contains
      subroutine work
!$omp parallel
         ! do work here
!$omp task
         tp = tp + 1
         ! do work here
!$omp task
           ! do work here but don't modify tp
!$omp end task
         var = tp    ! value does not change after write above
!$omp end task
!$omp end parallel
      end subroutine
      end module
