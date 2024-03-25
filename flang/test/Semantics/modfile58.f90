! RUN: %python %S/test_modfile.py %s %flang_fc1

! Test that the implicit SAVE attribute (set
! for the equivalenced symbols) is not written
! into the mod file.
module implicit_save
  real dx,dy
  common /blk/ dx
  equivalence(dx,dy)
end module implicit_save

!Expect: implicit_save.mod
!moduleimplicit_save
!real(4)::dx
!real(4)::dy
!common/blk/dx
!equivalence(dx,dy)
!end
