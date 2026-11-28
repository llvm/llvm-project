! RUN: bbc %s -o - | FileCheck %s

! Test initialization of COMMON blocks where initialized EQUIVALENCE objects
! overlap other initialized members or aliases. The combined initial value
! built by semantics must be used so that the entries do not overlap.

! 1.0 and 2.0 as REAL(4) bit patterns: 1065353216 and 1073741824.

! CHECK-LABEL: fir.global @__BLNK__ <{alignment = 4 : i64}> : tuple<!fir.array<2xi32>> {
! CHECK-DAG:     %[[TWO:.*]] = arith.constant 1073741824 : i32
! CHECK-DAG:     %[[ONE:.*]] = arith.constant 1065353216 : i32
! CHECK:         %[[ZERO:.*]] = fir.zero_bits tuple<!fir.array<2xi32>>
! CHECK:         %[[UNDEF:.*]] = fir.undefined !fir.array<2xi32>
! CHECK:         %[[ARR1:.*]] = fir.insert_value %[[UNDEF]], %[[ONE]], [0 : index]
! CHECK:         %[[ARR2:.*]] = fir.insert_value %[[ARR1]], %[[TWO]], [1 : index]
! CHECK:         %[[INIT:.*]] = fir.insert_value %[[ZERO]], %[[ARR2]], [0 : index]
! CHECK:         fir.has_value %[[INIT]] : tuple<!fir.array<2xi32>>

! CHECK-LABEL: fir.global @b_ <{alignment = 4 : i64}> : tuple<!fir.array<2xi32>> {
! CHECK-DAG:     %[[TWO:.*]] = arith.constant 1073741824 : i32
! CHECK-DAG:     %[[ONE:.*]] = arith.constant 1065353216 : i32
! CHECK:         %[[ZERO:.*]] = fir.zero_bits tuple<!fir.array<2xi32>>
! CHECK:         %[[UNDEF:.*]] = fir.undefined !fir.array<2xi32>
! CHECK:         %[[ARR1:.*]] = fir.insert_value %[[UNDEF]], %[[ONE]], [0 : index]
! CHECK:         %[[ARR2:.*]] = fir.insert_value %[[ARR1]], %[[TWO]], [1 : index]
! CHECK:         %[[INIT:.*]] = fir.insert_value %[[ZERO]], %[[ARR2]], [0 : index]
! CHECK:         fir.has_value %[[INIT]] : tuple<!fir.array<2xi32>>

! CHECK-LABEL: fir.global @c1_ <{alignment = 4 : i64}> : tuple<!fir.array<2xi32>> {
! CHECK-DAG:     %[[TWO:.*]] = arith.constant 1073741824 : i32
! CHECK-DAG:     %[[ONE:.*]] = arith.constant 1065353216 : i32
! CHECK:         %[[ZERO:.*]] = fir.zero_bits tuple<!fir.array<2xi32>>
! CHECK:         %[[UNDEF:.*]] = fir.undefined !fir.array<2xi32>
! CHECK:         %[[ARR1:.*]] = fir.insert_value %[[UNDEF]], %[[ONE]], [0 : index]
! CHECK:         %[[ARR2:.*]] = fir.insert_value %[[ARR1]], %[[TWO]], [1 : index]
! CHECK:         %[[INIT:.*]] = fir.insert_value %[[ZERO]], %[[ARR2]], [0 : index]
! CHECK:         fir.has_value %[[INIT]] : tuple<!fir.array<2xi32>>

! CHECK-LABEL: fir.global @c2_ <{alignment = 4 : i64}> : tuple<!fir.array<2xi32>> {
! CHECK-DAG:     %[[TWO:.*]] = arith.constant 1073741824 : i32
! CHECK-DAG:     %[[ONE:.*]] = arith.constant 1065353216 : i32
! CHECK:         %[[ZERO:.*]] = fir.zero_bits tuple<!fir.array<2xi32>>
! CHECK:         %[[UNDEF:.*]] = fir.undefined !fir.array<2xi32>
! CHECK:         %[[ARR1:.*]] = fir.insert_value %[[UNDEF]], %[[ONE]], [0 : index]
! CHECK:         %[[ARR2:.*]] = fir.insert_value %[[ARR1]], %[[TWO]], [1 : index]
! CHECK:         %[[INIT:.*]] = fir.insert_value %[[ZERO]], %[[ARR2]], [0 : index]
! CHECK:         fir.has_value %[[INIT]] : tuple<!fir.array<2xi32>>

! CHECK-LABEL: fir.global @c3_ <{alignment = 4 : i64}> : tuple<i32, !fir.array<2xi32>> {
! CHECK-DAG:     %[[C3:.*]] = arith.constant 3 : i32
! CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : i32
! CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : i32
! CHECK:         %[[ZERO:.*]] = fir.zero_bits tuple<i32, !fir.array<2xi32>>
! CHECK:         %[[TUP1:.*]] = fir.insert_value %[[ZERO]], %[[C1]], [0 : index]
! CHECK:         %[[UNDEF:.*]] = fir.undefined !fir.array<2xi32>
! CHECK:         %[[ARR1:.*]] = fir.insert_value %[[UNDEF]], %[[C2]], [0 : index]
! CHECK:         %[[ARR2:.*]] = fir.insert_value %[[ARR1]], %[[C3]], [1 : index]
! CHECK:         %[[INIT:.*]] = fir.insert_value %[[TUP1]], %[[ARR2]], [1 : index]
! CHECK:         fir.has_value %[[INIT]] : tuple<i32, !fir.array<2xi32>>

! The equivalenced object that overlaps x2 is not initialized: the members
! are still initialized separately.
! CHECK-LABEL: fir.global @c4_ <{alignment = 4 : i64}> : tuple<f32, f32> {
! CHECK-DAG:     %[[TWO:.*]] = arith.constant 2.000000e+00 : f32
! CHECK-DAG:     %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:         %[[ZERO:.*]] = fir.zero_bits tuple<f32, f32>
! CHECK:         %[[TUP1:.*]] = fir.insert_value %[[ZERO]], %[[ONE]], [0 : index]
! CHECK:         %[[INIT:.*]] = fir.insert_value %[[TUP1]], %[[TWO]], [1 : index]
! CHECK:         fir.has_value %[[INIT]] : tuple<f32, f32>

! A member and a partially overlapping alias are initialized.
block data bd
  real x1, x2, z(2)
  common /b/ x1, x2
  equivalence (x1, z(1))
  data x1 /1.0/, z(2) /2.0/
end

! Same, outside of BLOCK DATA.
subroutine named_common
  real x1, x2, z(2)
  common /c1/ x1, x2
  equivalence (x1, z(1))
  data x1 /1.0/, z(2) /2.0/
end

! Same, with blank COMMON.
subroutine blank_common
  real x1, x2, z(2)
  common // x1, x2
  equivalence (x1, z(1))
  data x1 /1.0/, z(2) /2.0/
end

! Only overlapping aliases are initialized.
block data aliases_only
  real x1, x2, z(2), w
  common /c2/ x1, x2
  equivalence (x1, z(1)), (x2, w)
  data z(1) /1.0/, w /2.0/
end

! The combined initial value starts after the first member.
block data after_first_member
  integer i0, i1, i2, j(2)
  common /c3/ i0, i1, i2
  equivalence (i1, j(1))
  data i0 /1/, i1 /2/, j(2) /3/
end

block data members_only
  real x1, x2, z(2)
  common /c4/ x1, x2
  equivalence (x1, z(1))
  data x1 /1.0/, x2 /2.0/
end
