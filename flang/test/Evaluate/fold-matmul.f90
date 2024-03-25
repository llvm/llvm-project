! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of MATMUL()
module m
  integer, parameter :: ia(2,3) = reshape([1, 2, 2, 3, 3, 4], shape(ia))
  integer, parameter :: ib(3,2) = reshape([1, 2, 3, 2, 3, 4], shape(ib))
  integer, parameter :: ix(*) = [1, 2]
  integer, parameter :: iy(*) = [1, 2, 3]
  integer, parameter :: iab(*,*) = matmul(ia, ib)
  integer, parameter :: ixa(*) = matmul(ix, ia)
  integer, parameter :: iay(*) = matmul(ia, iy)
  logical, parameter :: test_iab = all([iab] == [14, 20, 20, 29])
  logical, parameter :: test_ixa = all(ixa == [5, 8, 11])
  logical, parameter :: test_iay = all(iay == [14, 20])

  real, parameter :: ra(*,*) = ia
  real, parameter :: rb(*,*) = ib
  real, parameter :: rx(*) = ix
  real, parameter :: ry(*) = iy
  real, parameter :: rab(*,*) = matmul(ra, rb)
  real, parameter :: rxa(*) = matmul(rx, ra)
  real, parameter :: ray(*) = matmul(ra, ry)
  logical, parameter :: test_rab = all(rab == iab)
  logical, parameter :: test_rxa = all(rxa == ixa)
  logical, parameter :: test_ray = all(ray == iay)

  complex, parameter :: za(*,*) = cmplx(ra, -1.)
  complex, parameter :: zb(*,*) = cmplx(rb, -1.)
  complex, parameter :: zx(*) = cmplx(rx, -1.)
  complex, parameter :: zy(*) = cmplx(ry, -1.)
  complex, parameter :: zab(*,*) = matmul(za, zb)
  complex, parameter :: zxa(*) = matmul(zx, za)
  complex, parameter :: zay(*) = matmul(za, zy)
  logical, parameter :: test_zab = all([zab] == [(11,-12),(17,-15),(17,-15),(26,-18)])
  logical, parameter :: test_zxa = all(zxa == [(3,-6),(6,-8),(9,-10)])
  logical, parameter :: test_zay = all(zay == [(11,-12),(17,-15)])

  logical, parameter :: la(16, 4) = reshape([((iand(shiftr(j,k),1)/=0, j=0,15), k=0,3)], shape(la))
  logical, parameter :: lb(4, 16) = transpose(la)
  logical, parameter :: lab(16, 16) = matmul(la, lb)
  logical, parameter :: test_lab = all([lab] .eqv. [((iand(k,j)/=0, k=0,15), j=0,15)])
end
