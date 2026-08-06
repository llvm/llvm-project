! RUN: %python %S/test_errors.py %s %flang_fc1
! Test that a zero-size CHARACTER in a COMMON block does not trigger a false
! "cannot backward-extend COMMON block" error when it is referenced in an
! EQUIVALENCE association.
!
! Block /blk/ layout:
!   offset  0 : i8   (integer(8), 8 bytes)
!   offset  8 : zc0  (character(0), 0 bytes)
!
! equivalence(c8(5:8), zc0) places c8(1) at offset 4 (= 8 - 4 bytes before zc0).
! Before the fix (compute-offsets.cpp), zc0 had offset 0, so dep.offset (4)
! > symbol.offset() (0) triggered the backward-extend error falsely.
! After the fix, zc0.offset() == 8 so no backward extension occurs and no
! error should be emitted.

subroutine p09
  integer(8) :: i8
  character(0) :: zc0
  character(8) :: c8
  common /blk/ i8, zc0
  equivalence (c8(5:8), zc0)
  call use(c8, i8)
end subroutine
