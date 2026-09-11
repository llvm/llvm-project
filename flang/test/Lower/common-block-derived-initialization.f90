! Test lowering of a DATA-initialized common block, including a derived-type
! (structure-constructor) member, through the fir.global initializer region.
!
! Initial values are lowered via ConvertConstant. This pins the common-block
! initialization path (which reroutes every member's initial value through the
! same constant-lowering entry point) for a scalar, a real, and a derived-type
! member built from a structure constructor. DATA values are never
! parenthesized, so no fir.no_reassoc is expected.

! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s --implicit-check-not=fir.no_reassoc

block data bd
  type sq
    sequence
    integer :: n
  end type
  integer :: cbi
  real :: cbr
  type(sq) :: cbx
  common /blk/ cbi, cbr, cbx
  data cbi /42/, cbr /3.5/, cbx /sq(9)/
end block data

! CHECK-LABEL: fir.global @blk_
! CHECK-SAME:    tuple<i32, f32, !fir.type<_QTsq,sequence{n:i32}>>
! CHECK:         %[[AGG0:.*]] = fir.zero_bits tuple<i32, f32, !fir.type<_QTsq,sequence{n:i32}>>
! CHECK:         %[[C42:.*]] = arith.constant 42 : i32
! CHECK:         %[[AGG1:.*]] = fir.insert_value %[[AGG0]], %[[C42]], [0 : index]
! CHECK:         %[[CR:.*]] = arith.constant 3.500000e+00 : f32
! CHECK:         %[[AGG2:.*]] = fir.insert_value %[[AGG1]], %[[CR]], [1 : index]
! CHECK:         %[[SQ0:.*]] = fir.undefined !fir.type<_QTsq,sequence{n:i32}>
! CHECK:         %[[C9:.*]] = arith.constant 9 : i32
! CHECK:         %[[SQ1:.*]] = fir.insert_value %[[SQ0]], %[[C9]], ["n", !fir.type<_QTsq,sequence{n:i32}>]
! CHECK:         %[[AGG3:.*]] = fir.insert_value %[[AGG2]], %[[SQ1]], [2 : index]
! CHECK:         fir.has_value %[[AGG3]] : tuple<i32, f32, !fir.type<_QTsq,sequence{n:i32}>>
