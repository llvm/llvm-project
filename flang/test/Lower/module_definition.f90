! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test lowering of module that defines data that is otherwise not used
! in this file.

! Module m1 defines simple data
module m1
  real :: x
  integer :: y(100)
end module
! CHECK: fir.global @_QMm1Ex : f32
! CHECK: fir.global @_QMm1Ey : !fir.array<100xi32>

! Module modEq1 defines data that is equivalenced and not used in this
! file.
module modEq1
  ! Equivalence, no initialization
  real :: x1(10), x2(10), x3(10) 
  ! Equivalence with initialization
  real :: y1 = 42.
  real :: y2(10)
  equivalence (x1(1), x2(5), x3(10)), (y1, y2(5))
end module
! CHECK-LABEL: fir.global @_QMmodeq1Ex1 : tuple<!fir.array<36xi8>, !fir.array<40xi8>> {
  ! CHECK: %[[undef:.*]] = fir.undefined tuple<!fir.array<36xi8>, !fir.array<40xi8>>
  ! CHECK: fir.has_value %[[undef]] : tuple<!fir.array<36xi8>, !fir.array<40xi8>>
! CHECK-LABEL: fir.global @_QMmodeq1Ey1 : tuple<!fir.array<16xi8>, f32, !fir.array<20xi8>> {
  ! CHECK: %[[undef:.*]] = fir.undefined tuple<!fir.array<16xi8>, f32, !fir.array<20xi8>>
  ! CHECK: %[[cst:.*]] = constant 4.200000e+01 : f32
  ! CHECK: %[[init:.*]] = fir.insert_value %[[undef]], %[[cst]], %c1{{.*}} : (tuple<!fir.array<16xi8>, f32, !fir.array<20xi8>>, f32, index) -> tuple<!fir.array<16xi8>, f32, !fir.array<20xi8>>
  ! CHECK: fir.has_value %[[init]] : tuple<!fir.array<16xi8>, f32, !fir.array<20xi8>>

! Module defines variable in common block without initializer
module modCommonNoInit1
  ! Module variable is in blank common
  real :: x_blank
  common // x_blank
  ! Module variable is in named common, no init
  real :: x_named1
  common /named1/ x_named1
end module
! CHECK-LABEL: fir.global common @_QB(dense<0> : vector<4xi8>) : !fir.array<4xi8>
! CHECK-LABEL: fir.global common @_QBnamed1(dense<0> : vector<4xi8>) : !fir.array<4xi8>

! Module defines variable in common block with initialization
module modCommonInit1
  integer :: i_named2 = 42
  common /named2/ i_named2
end module
! CHECK-LABEL: fir.global @_QBnamed2 : tuple<i32> {
  ! CHECK: %[[init:.*]] = fir.insert_value %{{.*}}, %c42{{.*}}, %c0{{.*}} : (tuple<i32>, i32, index) -> tuple<i32>
  ! CHECK: fir.has_value %[[init]] : tuple<i32>
