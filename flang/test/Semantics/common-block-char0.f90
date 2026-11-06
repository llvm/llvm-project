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

subroutine p09
  integer(8) :: i8
  character(0) :: zc0
  character(8) :: c8
  common /blk/ i8, zc0
  equivalence (c8(5:8), zc0)
  call use(c8, i8)
end subroutine

! A genuine backward extension through a zero-size member must still error.
! Block /blk2/ layout:
!   offset  0 : zc0  (character(0), 0 bytes)
!   offset  0 : i8   (integer(8), 8 bytes)
!
! equivalence(c8(5:8), zc0) would place c8(1) at offset -4 -- before the
! block base -- which is a true backward extension and must be rejected.
subroutine backward
  integer(8) :: i8
  character(0) :: zc0
  character(8) :: c8
  !ERROR: 'zc0' cannot backward-extend COMMON block /blk2/ via EQUIVALENCE with 'c8'
  common /blk2/ zc0, i8
  equivalence (c8(5:8), zc0)
end subroutine
