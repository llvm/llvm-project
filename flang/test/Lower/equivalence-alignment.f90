! RUN: bbc -emit-hlfir -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! CHECK-LABEL: func.func @_QPalign4()
subroutine align4
  integer :: i
  real :: r
  equivalence (i, r)
  ! CHECK: fir.alloca !fir.array<4xi8> <{alignment = 4 : i64, uniq_name = "_QFalign4Ei"}>
end subroutine align4

! The strictest member sets the alignment, not the one listed first.
! CHECK-LABEL: func.func @_QPalign8()
subroutine align8
  integer :: i
  real(8) :: d
  equivalence (i, d)
  ! CHECK: fir.alloca !fir.array<8xi8> <{alignment = 8 : i64, uniq_name = "_QFalign8Ed"}>
end subroutine align8

! The block stays 8-byte aligned even though its size is not a multiple of 8.
! CHECK-LABEL: func.func @_QPalign8_odd_size()
subroutine align8_odd_size
  real(8) :: d(1)
  real :: r(3)
  equivalence (d(1), r(1))
  ! CHECK: fir.alloca !fir.array<12xi8> <{alignment = 8 : i64, uniq_name = "_QFalign8_odd_sizeEd"}>
end subroutine align8_odd_size

! CHECK-LABEL: func.func @_QPalign_char()
subroutine align_char
  character(len=4) :: c
  character(len=2) :: d
  equivalence (c, d)
  ! CHECK: fir.alloca !fir.array<4xi8> <{alignment = 1 : i64, uniq_name = "_QFalign_charEc"}>
end subroutine align_char
