! @@name:	tasking.14f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
subroutine foo()
integer i
!$omp task if(.FALSE.) ! This task is undeferred
!$omp task             ! This task is a regular task
  do i = 1, 3
    !$omp task             ! This task is a regular task
      call bar()
    !$omp end task
  enddo
!$omp end task
!$omp end task
!$omp task final(.TRUE.) ! This task is a regular task
!$omp task               ! This task is included
  do i = 1, 3
    !$omp task               ! This task is also included
     call bar()
    !$omp end task
  enddo
!$omp end task
!$omp end task
end subroutine
