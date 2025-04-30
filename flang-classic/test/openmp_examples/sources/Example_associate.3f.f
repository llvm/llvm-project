! @@name:	associate.3f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	yes
! @@expect:	success
program example
  integer :: v
  v = 15
associate(u => v)
!$omp parallel private(v)
  v = -1
  print *, v               ! private v=-1
  print *, u               ! original v=15
!$omp end parallel
end associate
end program
