! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test lowering of module that defines data that is otherwise not used
! in this file.

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
  ! CHECK: %[[init:.*]] = fir.insert_value %{{.*}}, %c42{{.*}}, [0 : index] : (tuple<i32>, i32) -> tuple<i32>
  ! CHECK: fir.has_value %[[init]] : tuple<i32>

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
! CHECK-LABEL: fir.global @_QMmodeq1Ex1 : !fir.array<76xi8>
! CHECK-LABEL: fir.global @_QMmodeq1Ey1 : !fir.array<10xi32> {
  ! CHECK: %[[undef:.*]] = fir.undefined !fir.array<10xi32>
  ! CHECK: %[[v1:.*]] = fir.insert_on_range %0, %c0{{.*}} from (0) to (3) : (!fir.array<10xi32>, i32) -> !fir.array<10xi32>
  ! CHECK: %[[v2:.*]] = fir.insert_value %1, %c1109917696{{.*}}, [4 : index] : (!fir.array<10xi32>, i32) -> !fir.array<10xi32>
  ! CHECK: %[[v3:.*]] = fir.insert_on_range %2, %c0{{.*}} from (5) to (9) : (!fir.array<10xi32>, i32) -> !fir.array<10xi32>
  ! CHECK: fir.has_value %[[v3]] : !fir.array<10xi32>

! Test defining two module variables whose initializers depend on each others
! addresses.
module global_init_depending_on_each_other_address
  type a
    type(b), pointer :: pb
  end type
  type b
    type(a), pointer :: pa
  end type
  type(a), target :: xa
  type(b), target :: xb
  data xa, xb/a(xb), b(xa)/
end module
! CHECK-LABEL: fir.global @_QMglobal_init_depending_on_each_other_addressExb
  ! CHECK: fir.address_of(@_QMglobal_init_depending_on_each_other_addressExa)
! CHECK-LABEL: fir.global @_QMglobal_init_depending_on_each_other_addressExa
  ! CHECK: fir.address_of(@_QMglobal_init_depending_on_each_other_addressExb)
