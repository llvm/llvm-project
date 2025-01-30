! Test lowering of RESHAPE intrinsic to HLFIR
! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

module types
  type t
  end type t
end module types

subroutine reshape_test(x, source, pd, sh, ord)
  integer :: x(:,:)
  integer :: source(:,:,:)
  integer :: pd(:,:,:)
  integer :: sh(2)
  integer :: ord(2)
  x = reshape(source, sh, pd, ord)
end subroutine reshape_test
! CHECK-LABEL:   func.func @_QPreshape_test(
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFreshape_testEord"} : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>)
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFreshape_testEpd"} : (!fir.box<!fir.array<?x?x?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?x?xi32>>, !fir.box<!fir.array<?x?x?xi32>>)
! CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFreshape_testEsh"} : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>)
! CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFreshape_testEsource"} : (!fir.box<!fir.array<?x?x?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?x?xi32>>, !fir.box<!fir.array<?x?x?xi32>>)
! CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFreshape_testEx"} : (!fir.box<!fir.array<?x?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?x?xi32>>)
! CHECK:           %[[VAL_15:.*]] = hlfir.reshape %[[VAL_13]]#0 %[[VAL_12]]#0 pad %[[VAL_9]]#0 order %[[VAL_8]]#0 : (!fir.box<!fir.array<?x?x?xi32>>, !fir.ref<!fir.array<2xi32>>, !fir.box<!fir.array<?x?x?xi32>>, !fir.ref<!fir.array<2xi32>>) -> !hlfir.expr<?x?xi32>
! CHECK:           hlfir.assign %[[VAL_15]] to %[[VAL_14]]#0 : !hlfir.expr<?x?xi32>, !fir.box<!fir.array<?x?xi32>>
! CHECK:           hlfir.destroy %[[VAL_15]] : !hlfir.expr<?x?xi32>

subroutine reshape_test_noorder(x, source, pd, sh)
  integer :: x(:,:)
  integer :: source(:,:,:)
  integer :: pd(:,:,:)
  integer :: sh(2)
  x = reshape(source, sh, pd)
end subroutine reshape_test_noorder
! CHECK-LABEL:   func.func @_QPreshape_test_noorder(
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFreshape_test_noorderEpd"} : (!fir.box<!fir.array<?x?x?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?x?xi32>>, !fir.box<!fir.array<?x?x?xi32>>)
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFreshape_test_noorderEsh"} : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>)
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFreshape_test_noorderEsource"} : (!fir.box<!fir.array<?x?x?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?x?xi32>>, !fir.box<!fir.array<?x?x?xi32>>)
! CHECK:           %[[VAL_11:.*]] = hlfir.reshape %[[VAL_9]]#0 %[[VAL_8]]#0 pad %[[VAL_5]]#0 : (!fir.box<!fir.array<?x?x?xi32>>, !fir.ref<!fir.array<2xi32>>, !fir.box<!fir.array<?x?x?xi32>>) -> !hlfir.expr<?x?xi32>

subroutine reshape_test_nopad(x, source, sh, ord)
  integer :: x(:,:)
  integer :: source(:,:,:)
  integer :: sh(2)
  integer :: ord(2)
  x = reshape(source, sh, ORDER=ord)
end subroutine reshape_test_nopad
! CHECK-LABEL:   func.func @_QPreshape_test_nopad(
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFreshape_test_nopadEord"} : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>)
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFreshape_test_nopadEsh"} : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>)
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFreshape_test_nopadEsource"} : (!fir.box<!fir.array<?x?x?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?x?xi32>>, !fir.box<!fir.array<?x?x?xi32>>)
! CHECK:           %[[VAL_13:.*]] = hlfir.reshape %[[VAL_11]]#0 %[[VAL_10]]#0 order %[[VAL_7]]#0 : (!fir.box<!fir.array<?x?x?xi32>>, !fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>) -> !hlfir.expr<?x?xi32>
  
subroutine test_reshape_optional1(pad, order, source, shape)
  real, pointer :: pad(:, :)
  integer, pointer :: order(:)
  real :: source(:, :, :)
  integer :: shape(4)
  print *, reshape(source=source, shape=shape, pad=pad, order=order)
end subroutine test_reshape_optional1
! CHECK-LABEL:   func.func @_QPtest_reshape_optional1(
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare {{.*}}{fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_reshape_optional1Eorder"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare {{.*}}{fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_reshape_optional1Epad"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>)
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFtest_reshape_optional1Eshape"} : (!fir.ref<!fir.array<4xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<4xi32>>, !fir.ref<!fir.array<4xi32>>)
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFtest_reshape_optional1Esource"} : (!fir.box<!fir.array<?x?x?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?x?xf32>>, !fir.box<!fir.array<?x?x?xf32>>)
! CHECK:           %[[VAL_16:.*]] = fir.load %[[VAL_6]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
! CHECK:           %[[VAL_17:.*]] = fir.box_addr %[[VAL_16]] : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>) -> !fir.ptr<!fir.array<?x?xf32>>
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (!fir.ptr<!fir.array<?x?xf32>>) -> i64
! CHECK:           %[[VAL_19:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_20:.*]] = arith.cmpi ne, %[[VAL_18]], %[[VAL_19]] : i64
! CHECK:           %[[VAL_21:.*]] = fir.load %[[VAL_5]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_22:.*]] = fir.box_addr %[[VAL_21]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (!fir.ptr<!fir.array<?xi32>>) -> i64
! CHECK:           %[[VAL_24:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_25:.*]] = arith.cmpi ne, %[[VAL_23]], %[[VAL_24]] : i64
! CHECK:           %[[VAL_26:.*]] = fir.load %[[VAL_6]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
! CHECK:           %[[VAL_27:.*]] = fir.absent !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
! CHECK:           %[[VAL_28:.*]] = arith.select %[[VAL_20]], %[[VAL_26]], %[[VAL_27]] : !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
! CHECK:           %[[VAL_29:.*]] = fir.load %[[VAL_5]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_30:.*]] = fir.absent !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:           %[[VAL_31:.*]] = arith.select %[[VAL_25]], %[[VAL_29]], %[[VAL_30]] : !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:           %[[VAL_32:.*]] = hlfir.reshape %[[VAL_10]]#0 %[[VAL_9]]#0 pad %[[VAL_28]] order %[[VAL_31]] : (!fir.box<!fir.array<?x?x?xf32>>, !fir.ref<!fir.array<4xi32>>, !fir.box<!fir.ptr<!fir.array<?x?xf32>>>, !fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !hlfir.expr<?x?x?x?xf32>

subroutine test_reshape_optional2(pad, order, source, shape)
  real, optional :: pad(:, :)
  integer, pointer, optional :: order(:)
  real :: source(:, :, :)
  integer :: shape(4)
  print *, reshape(source=source, shape=shape, pad=pad, order=order)
end subroutine test_reshape_optional2
! CHECK-LABEL:   func.func @_QPtest_reshape_optional2(
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare {{.*}}{fortran_attrs = #fir.var_attrs<optional, pointer>, uniq_name = "_QFtest_reshape_optional2Eorder"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare {{.*}}{fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFtest_reshape_optional2Epad"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFtest_reshape_optional2Eshape"} : (!fir.ref<!fir.array<4xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<4xi32>>, !fir.ref<!fir.array<4xi32>>)
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFtest_reshape_optional2Esource"} : (!fir.box<!fir.array<?x?x?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?x?xf32>>, !fir.box<!fir.array<?x?x?xf32>>)
! CHECK:           %[[VAL_16:.*]] = fir.is_present %[[VAL_6]]#0 : (!fir.box<!fir.array<?x?xf32>>) -> i1
! CHECK:           %[[VAL_17:.*]] = fir.load %[[VAL_5]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_18:.*]] = fir.box_addr %[[VAL_17]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (!fir.ptr<!fir.array<?xi32>>) -> i64
! CHECK:           %[[VAL_20:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_21:.*]] = arith.cmpi ne, %[[VAL_19]], %[[VAL_20]] : i64
! CHECK:           %[[VAL_22:.*]] = fir.absent !fir.box<!fir.array<?x?xf32>>
! CHECK:           %[[VAL_23:.*]] = arith.select %[[VAL_16]], %[[VAL_6]]#1, %[[VAL_22]] : !fir.box<!fir.array<?x?xf32>>
! CHECK:           %[[VAL_24:.*]] = fir.load %[[VAL_5]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_25:.*]] = fir.absent !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:           %[[VAL_26:.*]] = arith.select %[[VAL_21]], %[[VAL_24]], %[[VAL_25]] : !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:           %[[VAL_27:.*]] = hlfir.reshape %[[VAL_10]]#0 %[[VAL_9]]#0 pad %[[VAL_23]] order %[[VAL_26]] : (!fir.box<!fir.array<?x?x?xf32>>, !fir.ref<!fir.array<4xi32>>, !fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !hlfir.expr<?x?x?x?xf32>

subroutine test_reshape_shape_expr(source, shape)
  integer :: source(:), shape(2)
  print *, reshape(source, shape + 1)
end subroutine test_reshape_shape_expr
! CHECK-LABEL:   func.func @_QPtest_reshape_shape_expr(
! CHECK:           %[[VAL_13:.*]] = hlfir.elemental
! CHECK:           %[[VAL_18:.*]] = hlfir.reshape %{{.*}} %[[VAL_13]] : (!fir.box<!fir.array<?xi32>>, !hlfir.expr<2xi32>) -> !hlfir.expr<?x?xi32>

subroutine test_reshape_polymorphic1(source, shape)
  use types
  class(t), allocatable :: source(:)
  integer :: shape(1)
  source = reshape(source, shape)
end subroutine test_reshape_polymorphic1
! CHECK-LABEL:   func.func @_QPtest_reshape_polymorphic1(
! CHECK:           hlfir.reshape %{{.*}} %{{.*}} : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?x!fir.type<_QMtypesTt>?>

subroutine test_reshape_polymorphic2(source, shape, pad)
  use types
  class(t), allocatable :: source(:)
  type(t) :: pad(:)
  integer :: shape(1)
  source = reshape(source, shape, pad)
end subroutine test_reshape_polymorphic2
! CHECK-LABEL:   func.func @_QPtest_reshape_polymorphic2(
! CHECK:           hlfir.reshape %{{.*}} %{{.*}} pad %{{.*}} : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, !fir.ref<!fir.array<1xi32>>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>) -> !hlfir.expr<?x!fir.type<_QMtypesTt>?>

subroutine test_reshape_polymorphic3(source, shape, pad)
  use types
  type(t) :: source(:)
  class(t) :: pad(:)
  integer :: shape(1)
  source = reshape(source, shape, pad)
end subroutine test_reshape_polymorphic3
! CHECK-LABEL:   func.func @_QPtest_reshape_polymorphic3(
! CHECK:           hlfir.reshape %{{.*}} %{{.*}} pad %{{.*}} : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>, !fir.ref<!fir.array<1xi32>>, !fir.class<!fir.array<?x!fir.type<_QMtypesTt>>>) -> !hlfir.expr<?x!fir.type<_QMtypesTt>>
