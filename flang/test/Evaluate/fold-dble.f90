! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of DBLE()
module ft_data
  integer nx, ny, nz
  parameter (nx=64, ny=64, nz=64, maxdim=64)
  double precision ntotal_f
  parameter (ntotal_f =  dble(nx)*ny*nz)
  logical, parameter :: test_dble_1 = ntotal_f == real(nx*ny*nz, kind(0.0D0))
  logical, parameter :: test_dble_2 = kind(dble(nx)) == kind(0.0D0)
end module ft_data
