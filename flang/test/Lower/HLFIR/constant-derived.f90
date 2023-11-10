! Test lowering of Constant<SomeDerived>.
! RUN: bbc -emit-hlfir -o - -I nowhere %s 2>&1 | FileCheck %s

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

subroutine test_constant_scalar_ptr_component()
  type myderived
    real, pointer :: x
    real, pointer :: y(:)
  end type
  real, target, save :: targ(100)
  call test(myderived(NULL(), targ(1:50:5)))
! CHECK-LABEL: func.func @_QPtest_constant_scalar_ptr_component() {
! CHECK:  fir.address_of(@[[CST_TARGET:_QQro._QFtest_constant_scalar_ptr_componentTmyderived..*]])
end subroutine

subroutine test_comp_ref()
  ! Test parent component value in an initial value structure constructor.
  type t1
    integer :: i
  end type
  type, extends(t1) :: t2
    integer :: j
  end type
  type(t2) :: x = t2(t1=t1(1), j=2)
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

! CHECK: fir.global internal @[[CST_TARGET]] constant :
! CHECK-SAME: !fir.type<[[DERIVED_2:_QFtest_constant_scalar_ptr_componentTmyderived{x:!fir.box<!fir.ptr<f32>>,y:!fir.box<!fir.ptr<!fir.array<\?xf32>>>}]]> {
! CHECK:   %[[VAL_0:.*]] = fir.undefined !fir.type<[[DERIVED_2]]>
! CHECK:   %[[VAL_1:.*]] = fir.field_index x, !fir.type<[[DERIVED_2]]>
! CHECK:   %[[VAL_2:.*]] = fir.zero_bits !fir.ptr<f32>
! CHECK:   %[[VAL_3:.*]] = fir.embox %[[VAL_2]] : (!fir.ptr<f32>) -> !fir.box<!fir.ptr<f32>>
! CHECK:   %[[VAL_4:.*]] = fir.insert_value %[[VAL_0]], %[[VAL_3]], ["x", !fir.type<[[DERIVED_2]]>] : (!fir.type<[[DERIVED_2]]>, !fir.box<!fir.ptr<f32>>) -> !fir.type<[[DERIVED_2]]>
! CHECK:   %[[VAL_5:.*]] = fir.field_index y, !fir.type<[[DERIVED_2]]>
! CHECK:   %[[VAL_6:.*]] = fir.address_of(@_QFtest_constant_scalar_ptr_componentEtarg) : !fir.ref<!fir.array<100xf32>>
! CHECK:   %[[VAL_7:.*]] = arith.constant 100 : index
! CHECK:   %[[VAL_8:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:   %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_6]](%[[VAL_8]])
! CHECK:   %[[VAL_11:.*]] = arith.constant 1 : index
! CHECK:   %[[VAL_13:.*]] = arith.constant 50 : index
! CHECK:   %[[VAL_15:.*]] = arith.constant 5 : index
! CHECK:   %[[VAL_16:.*]] = arith.constant 10 : index
! CHECK:   %[[VAL_17:.*]] = fir.shape %[[VAL_16]] : (index) -> !fir.shape<1>
! CHECK:   %[[VAL_18:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_11]]:%[[VAL_13]]:%[[VAL_15]])  shape %[[VAL_17]] : (!fir.ref<!fir.array<100xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<10xf32>>
! CHECK:   %[[VAL_19:.*]] = fir.rebox %[[VAL_18]] : (!fir.box<!fir.array<10xf32>>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:   %[[VAL_20:.*]] = fir.insert_value %[[VAL_4]], %[[VAL_19]], ["y", !fir.type<[[DERIVED_2]]>] : (!fir.type<[[DERIVED_2]]>, !fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.type<[[DERIVED_2]]>
! CHECK:   fir.has_value %[[VAL_20]] : !fir.type<[[DERIVED_2]]>
! CHECK: }

! CHECK-LABEL:   fir.global internal @_QFtest_comp_refEx : !fir.type<_QFtest_comp_refTt2{t1:!fir.type<_QFtest_comp_refTt1{i:i32}>,j:i32}> {
! CHECK:   %[[VAL_0:.*]] = fir.undefined !fir.type<_QFtest_comp_refTt2{t1:!fir.type<_QFtest_comp_refTt1{i:i32}>,j:i32}>
! CHECK:   %[[VAL_1:.*]] = fir.field_index t1, !fir.type<_QFtest_comp_refTt2{t1:!fir.type<_QFtest_comp_refTt1{i:i32}>,j:i32}>
! CHECK:   %[[VAL_2:.*]] = fir.undefined !fir.type<_QFtest_comp_refTt1{i:i32}>
! CHECK:   %[[VAL_3:.*]] = fir.field_index i, !fir.type<_QFtest_comp_refTt1{i:i32}>
! CHECK:   %[[VAL_4:.*]] = arith.constant 1 : i32
! CHECK:   %[[VAL_5:.*]] = fir.insert_value %[[VAL_2]], %[[VAL_4]], ["i", !fir.type<_QFtest_comp_refTt1{i:i32}>] : (!fir.type<_QFtest_comp_refTt1{i:i32}>, i32) -> !fir.type<_QFtest_comp_refTt1{i:i32}>
! CHECK:   %[[VAL_6:.*]] = fir.insert_value %[[VAL_0]], %[[VAL_5]], ["t1", !fir.type<_QFtest_comp_refTt2{t1:!fir.type<_QFtest_comp_refTt1{i:i32}>,j:i32}>] : (!fir.type<_QFtest_comp_refTt2{t1:!fir.type<_QFtest_comp_refTt1{i:i32}>,j:i32}>, !fir.type<_QFtest_comp_refTt1{i:i32}>) -> !fir.type<_QFtest_comp_refTt2{t1:!fir.type<_QFtest_comp_refTt1{i:i32}>,j:i32}>
! CHECK:   %[[VAL_7:.*]] = fir.field_index j, !fir.type<_QFtest_comp_refTt2{t1:!fir.type<_QFtest_comp_refTt1{i:i32}>,j:i32}>
! CHECK:   %[[VAL_8:.*]] = arith.constant 2 : i32
! CHECK:   %[[VAL_9:.*]] = fir.insert_value %[[VAL_6]], %[[VAL_8]], ["j", !fir.type<_QFtest_comp_refTt2{t1:!fir.type<_QFtest_comp_refTt1{i:i32}>,j:i32}>] : (!fir.type<_QFtest_comp_refTt2{t1:!fir.type<_QFtest_comp_refTt1{i:i32}>,j:i32}>, i32) -> !fir.type<_QFtest_comp_refTt2{t1:!fir.type<_QFtest_comp_refTt1{i:i32}>,j:i32}>
! CHECK:   fir.has_value %[[VAL_9]] : !fir.type<_QFtest_comp_refTt2{t1:!fir.type<_QFtest_comp_refTt1{i:i32}>,j:i32}>
! CHECK: }
