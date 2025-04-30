! @@name:	affinity.2f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
subroutine foo
!$omp parallel num_threads(16) proc_bind(spread)
      call work()
!$omp end parallel
end subroutine
