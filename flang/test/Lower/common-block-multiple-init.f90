! RUN: bbc %s -o - | FileCheck %s

! Test that a duplicate (identical-valued) initialization of a named
! COMMON block across program units still lowers correctly, embedding the
! (shared) value once. Only duplicate initializations reach lowering: a
! disjoint or otherwise conflicting initialization across appearances is
! rejected earlier, in semantics -- see
! flang/test/Semantics/common-block-multiple-init.f90 and
! flang/docs/Extensions.md.

! CHECK-LABEL: fir.global @blk_ <{alignment = 4 : i64}> : !fir.type<_QBTblk_<{m0:i32,m1:!fir.array<4xi8>}>> {
! CHECK:  %[[val:.*]] = arith.constant 111 : i32
! CHECK:  %[[undef:.*]] = fir.zero_bits !fir.type<_QBTblk_<{m0:i32,m1:!fir.array<4xi8>}>>
! CHECK:  %[[init:.*]] = fir.insert_value %[[undef]], %[[val]], ["m0", !fir.type<_QBTblk_<{m0:i32,m1:!fir.array<4xi8>}>>] : (!fir.type<_QBTblk_<{m0:i32,m1:!fir.array<4xi8>}>>, i32) -> !fir.type<_QBTblk_<{m0:i32,m1:!fir.array<4xi8>}>>
! CHECK-NOT: fir.insert_value
! CHECK:  fir.has_value %[[init]] : !fir.type<_QBTblk_<{m0:i32,m1:!fir.array<4xi8>}>>

subroutine first
  integer :: i, j
  common /blk/ i, j
  data i /111/
end subroutine

subroutine second
  integer :: i, j
  common /blk/ i, j
  data i /111/
end subroutine
