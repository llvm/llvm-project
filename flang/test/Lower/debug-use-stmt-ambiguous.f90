! RUN: %flang_fc1 -emit-hlfir -debug-info-kind=standalone %s -o -

! The test checks that ambigious use statements don't cause a build failure
! when generation of fir.use_stmt is enabled.
module module_1st
  integer :: mpi_integer = 1
  integer :: mpi = 2, ssss = 3
end module module_1st

module module_2nd
  integer :: dp, ssss
end module module_2nd

module module_3rd
  use module_1st
  use module_2nd
  integer :: itmp
end module module_3rd

program test
  use module_3rd, ONLY: ssss
  use module_3rd, ONLY:
  use module_1st
  implicit none

  ! Use non-ambiguous symbols
  if (mpi_integer .ne. 1) print *, 'ng'
  if (mpi .ne. 2) print *, 'ng'
  print *, 'pass'
end program test
