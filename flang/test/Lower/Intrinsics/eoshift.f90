! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPeoshift_test1(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.array<3x!fir.logical<4>>>{{.*}}, %[[ARG1:.*]]: !fir.ref<i32>{{.*}})
subroutine eoshift_test1(arr, shift)
    logical, dimension(3) :: arr, res
    integer :: shift
  ! CHECK: %[[DS:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[ARR_DECL:.*]]:2 = hlfir.declare %[[ARG0]]
  ! CHECK: %[[RES_ALLOC:.*]] = fir.alloca !fir.array<3x!fir.logical<4>>
  ! CHECK: %[[RES_DECL:.*]]:2 = hlfir.declare %[[RES_ALLOC]]
  ! CHECK: %[[SHIFT_DECL:.*]]:2 = hlfir.declare %[[ARG1]]
  ! CHECK: %[[EXPR:.*]] = hlfir.eoshift %[[ARR_DECL]]#0 %[[SHIFT_DECL]]#0 : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.ref<i32>) -> !hlfir.expr<3x!fir.logical<4>>
  ! CHECK: hlfir.assign %[[EXPR]] to %[[RES_DECL]]#0 : !hlfir.expr<3x!fir.logical<4>>, !fir.ref<!fir.array<3x!fir.logical<4>>>
  ! CHECK: hlfir.destroy %[[EXPR]] : !hlfir.expr<3x!fir.logical<4>>
    res = eoshift(arr, shift)
end subroutine eoshift_test1

! CHECK-LABEL: func.func @_QPeoshift_test2(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.array<3x3xi32>>{{.*}}, %[[ARG1:.*]]: !fir.ref<!fir.array<3xi32>>{{.*}}, %[[ARG2:.*]]: !fir.ref<i32>{{.*}}, %[[ARG3:.*]]: !fir.ref<i32>{{.*}})
subroutine eoshift_test2(arr, shift, bound, dim)
  integer, dimension(3,3) :: arr, res
  integer, dimension(3) :: shift
  integer :: bound, dim
  ! CHECK: %[[DS:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[ARR_DECL:.*]]:2 = hlfir.declare %[[ARG0]]
  ! CHECK: %[[BOUND_DECL:.*]]:2 = hlfir.declare %[[ARG2]]
  ! CHECK: %[[DIM_DECL:.*]]:2 = hlfir.declare %[[ARG3]]
  ! CHECK: %[[RES_ALLOC:.*]] = fir.alloca !fir.array<3x3xi32>
  ! CHECK: %[[RES_DECL:.*]]:2 = hlfir.declare %[[RES_ALLOC]]
  ! CHECK: %[[SHIFT_DECL:.*]]:2 = hlfir.declare %[[ARG1]]
  ! CHECK: %[[DIM_VAL:.*]] = fir.load %[[DIM_DECL]]#0 : !fir.ref<i32>
  ! CHECK: %[[EXPR:.*]] = hlfir.eoshift %[[ARR_DECL]]#0 %[[SHIFT_DECL]]#0 boundary %[[BOUND_DECL]]#0 dim %[[DIM_VAL]] : (!fir.ref<!fir.array<3x3xi32>>, !fir.ref<!fir.array<3xi32>>, !fir.ref<i32>, i32) -> !hlfir.expr<3x3xi32>
  ! CHECK: hlfir.assign %[[EXPR]] to %[[RES_DECL]]#0 : !hlfir.expr<3x3xi32>, !fir.ref<!fir.array<3x3xi32>>
  ! CHECK: hlfir.destroy %[[EXPR]] : !hlfir.expr<3x3xi32>
  res = eoshift(arr, shift, bound, dim)
end subroutine eoshift_test2

! CHECK-LABEL: func.func @_QPeoshift_test3(
! CHECK-SAME: %[[ARG0:.*]]: !fir.boxchar<1>{{.*}}, %[[ARG1:.*]]: !fir.ref<i32>{{.*}}, %[[ARG2:.*]]: !fir.ref<i32>{{.*}})
subroutine eoshift_test3(arr, shift, dim)
  character(4), dimension(3,3) :: arr, res
  integer :: shift, dim
  ! CHECK: %[[DS:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[UNBOX:.*]]:2 = fir.unboxchar %[[ARG0]]
  ! CHECK: %[[ARR_DECL:.*]]:2 = hlfir.declare %{{.*}}({{.*}}) typeparams %{{.*}} dummy_scope %[[DS]]
  ! CHECK: %[[DIM_DECL:.*]]:2 = hlfir.declare %[[ARG2]]
  ! CHECK: %[[RES_ALLOC:.*]] = fir.alloca !fir.array<3x3x!fir.char<1,4>>
  ! CHECK: %[[RES_DECL:.*]]:2 = hlfir.declare %[[RES_ALLOC]]({{.*}}) typeparams %{{.*}}
  ! CHECK: %[[SHIFT_DECL:.*]]:2 = hlfir.declare %[[ARG1]]
  ! CHECK: %[[DIM_VAL:.*]] = fir.load %[[DIM_DECL]]#0 : !fir.ref<i32>
  ! CHECK: %[[EXPR:.*]] = hlfir.eoshift %[[ARR_DECL]]#0 %[[SHIFT_DECL]]#0 dim %[[DIM_VAL]] : (!fir.ref<!fir.array<3x3x!fir.char<1,4>>>, !fir.ref<i32>, i32) -> !hlfir.expr<3x3x!fir.char<1,4>>
  ! CHECK: hlfir.assign %[[EXPR]] to %[[RES_DECL]]#0 : !hlfir.expr<3x3x!fir.char<1,4>>, !fir.ref<!fir.array<3x3x!fir.char<1,4>>>
  ! CHECK: hlfir.destroy %[[EXPR]] : !hlfir.expr<3x3x!fir.char<1,4>>
  res = eoshift(arr, SHIFT=shift, DIM=dim)
end subroutine eoshift_test3

! CHECK-LABEL: func.func @_QPeoshift_test_dynamic_optional(
! CHECK-SAME:  %[[ARG0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[ARG1:.*]]: !fir.ref<i32>{{.*}}, %[[ARG2:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}} {fir.bindc_name = "boundary", fir.optional})
subroutine eoshift_test_dynamic_optional(array, shift, boundary)
  integer :: array(:, :)
  integer :: shift
  integer, optional :: boundary(10)
  ! CHECK: %[[ARRAY_DECL:.*]]:2 = hlfir.declare %[[ARG0]]
  ! CHECK: %[[BOUND_DECL:.*]]:2 = hlfir.declare %[[ARG2]]({{.*}})
  ! CHECK: %[[SHIFT_DECL:.*]]:2 = hlfir.declare %[[ARG1]]
  ! CHECK: %[[EXPR:.*]] = hlfir.eoshift %[[ARRAY_DECL]]#0 %[[SHIFT_DECL]]#0 boundary %{{.*}} : (!fir.box<!fir.array<?x?xi32>>, !fir.ref<i32>, !fir.box<!fir.array<10xi32>>) -> !hlfir.expr<?x?xi32>
  ! CHECK: hlfir.associate %[[EXPR]]
  ! CHECK: fir.call @_QPnext(
  ! CHECK: hlfir.destroy %[[EXPR]]
  call next(eoshift(array, shift, boundary))
end subroutine
