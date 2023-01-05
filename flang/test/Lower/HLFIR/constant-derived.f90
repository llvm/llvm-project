! Test lowering of Constant<SomeDerived>.
! TODO: remove "-I nowhere" once derived type descriptor can be lowered.
! RUN: bbc -hlfir -emit-fir -o - -I nowhere %s 2>&1 | FileCheck %s

subroutine test_constant_scalar()
  type myderived
    integer :: i
    integer :: j = 42
    real :: x(2)
    character(10) :: c
  end type
  print *, myderived(i=1, x=[2.,3.], c="hello")
! CHECK-LABEL: func.func @_QPtest_constant_scalar() {
! CHECK:  fir.address_of(@[[CST:_QQro._QFtest_constant_scalarTmyderived..*]])
end subroutine

! CHECK: fir.global internal @[[CST]] constant : !fir.type<[[DERIVED:_QFtest_constant_scalarTmyderived{i:i32,j:i32,x:!fir.array<2xf32>,c:!fir.char<1,10>}]]> {
! CHECK:   %[[VAL_0:.*]] = fir.undefined !fir.type<[[DERIVED]]>
! CHECK:   %[[VAL_1:.*]] = fir.field_index i, !fir.type<[[DERIVED]]>
! CHECK:   %[[VAL_2:.*]] = arith.constant 1 : i32
! CHECK:   %[[VAL_3:.*]] = fir.insert_value %[[VAL_0]], %[[VAL_2]], ["i", !fir.type<[[DERIVED]]>] : (!fir.type<[[DERIVED]]>, i32) -> !fir.type<[[DERIVED]]>
! CHECK:   %[[VAL_4:.*]] = fir.field_index j, !fir.type<[[DERIVED]]>
! CHECK:   %[[VAL_5:.*]] = arith.constant 42 : i32
! CHECK:   %[[VAL_6:.*]] = fir.insert_value %[[VAL_3]], %[[VAL_5]], ["j", !fir.type<[[DERIVED]]>] : (!fir.type<[[DERIVED]]>, i32) -> !fir.type<[[DERIVED]]>
! CHECK:   %[[VAL_7:.*]] = fir.field_index x, !fir.type<[[DERIVED]]>
! CHECK:   %[[VAL_8:.*]] = fir.undefined !fir.array<2xf32>
! CHECK:   %[[VAL_9:.*]] = arith.constant 2.000000e+00 : f32
! CHECK:   %[[VAL_10:.*]] = fir.insert_value %[[VAL_8]], %[[VAL_9]], [0 : index] : (!fir.array<2xf32>, f32) -> !fir.array<2xf32>
! CHECK:   %[[VAL_11:.*]] = arith.constant 3.000000e+00 : f32
! CHECK:   %[[VAL_12:.*]] = fir.insert_value %[[VAL_10]], %[[VAL_11]], [1 : index] : (!fir.array<2xf32>, f32) -> !fir.array<2xf32>
! CHECK:   %[[VAL_13:.*]] = arith.constant 2 : index
! CHECK:   %[[VAL_14:.*]] = fir.insert_value %[[VAL_6]], %[[VAL_12]], ["x", !fir.type<[[DERIVED]]>] : (!fir.type<[[DERIVED]]>, !fir.array<2xf32>) -> !fir.type<[[DERIVED]]>
! CHECK:   %[[VAL_15:.*]] = fir.field_index c, !fir.type<[[DERIVED]]>
! CHECK:   %[[VAL_16:.*]] = fir.string_lit "hello     "(10) : !fir.char<1,10>
! CHECK:   %[[VAL_17:.*]] = arith.constant 10 : index
! CHECK:   %[[VAL_18:.*]] = fir.insert_value %[[VAL_14]], %[[VAL_16]], ["c", !fir.type<[[DERIVED]]>] : (!fir.type<[[DERIVED]]>, !fir.char<1,10>) -> !fir.type<[[DERIVED]]>
! CHECK:   fir.has_value %[[VAL_18]] : !fir.type<[[DERIVED]]>
! CHECK: }
