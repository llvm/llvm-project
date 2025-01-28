! Test lowering of calls to interface with non pointer non allocatable
! assumed rank dummy arguments.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

module ifaces
  interface
    subroutine int_assumed_rank(y)
      integer :: y(..)
    end subroutine
    subroutine int_opt_assumed_rank(y)
      integer, optional :: y(..)
    end subroutine
    subroutine int_assumed_rank_bindc(y) bind(c)
      integer :: y(..)
    end subroutine
  end interface
end module

subroutine int_scalar_to_assumed_rank(x)
  use ifaces, only : int_assumed_rank
  integer :: x
  call int_assumed_rank(x)
end subroutine
! CHECK-LABEL:   func.func @_QPint_scalar_to_assumed_rank(
! CHECK-SAME:                                             %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFint_scalar_to_assumed_rankEx"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]] = fir.embox %[[VAL_1]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.box<i32>) -> !fir.box<!fir.array<*:i32>>
! CHECK:           fir.call @_QPint_assumed_rank(%[[VAL_3]]) fastmath<contract> : (!fir.box<!fir.array<*:i32>>) -> ()

subroutine int_scalar_to_assumed_rank_bindc(x)
  use ifaces, only : int_assumed_rank_bindc
  integer :: x
  call int_assumed_rank_bindc(x)
end subroutine
! CHECK-LABEL:   func.func @_QPint_scalar_to_assumed_rank_bindc(
! CHECK-SAME:                                                   %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFint_scalar_to_assumed_rank_bindcEx"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]] = fir.embox %[[VAL_1]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.box<i32>) -> !fir.box<!fir.array<*:i32>>
! CHECK:           fir.call @int_assumed_rank_bindc(%[[VAL_3]]) proc_attrs<bind_c> fastmath<contract> : (!fir.box<!fir.array<*:i32>>) -> ()

subroutine int_r1_to_assumed_rank(x)
  use ifaces, only : int_assumed_rank
  integer :: x(10)
  call int_assumed_rank(x)
end subroutine
! CHECK-LABEL:   func.func @_QPint_r1_to_assumed_rank(
! CHECK-SAME:                                         %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_2:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_2]]) dummy_scope %{{[0-9]+}} {uniq_name = "_QFint_r1_to_assumed_rankEx"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[VAL_4:.*]] = fir.embox %[[VAL_3]]#0(%[[VAL_2]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.box<!fir.array<10xi32>>) -> !fir.box<!fir.array<*:i32>>
! CHECK:           fir.call @_QPint_assumed_rank(%[[VAL_5]]) fastmath<contract> : (!fir.box<!fir.array<*:i32>>) -> ()

subroutine int_r4_to_assumed_rank(x)
  use ifaces, only : int_assumed_rank
  integer :: x(2,3,4,5)
  call int_assumed_rank(x)
end subroutine
! CHECK-LABEL:   func.func @_QPint_r4_to_assumed_rank(
! CHECK-SAME:                                         %[[VAL_0:.*]]: !fir.ref<!fir.array<2x3x4x5xi32>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_2:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_3:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_4:.*]] = arith.constant 5 : index
! CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]] : (index, index, index, index) -> !fir.shape<4>
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_5]]) dummy_scope %{{[0-9]+}} {uniq_name = "_QFint_r4_to_assumed_rankEx"} : (!fir.ref<!fir.array<2x3x4x5xi32>>, !fir.shape<4>, !fir.dscope) -> (!fir.ref<!fir.array<2x3x4x5xi32>>, !fir.ref<!fir.array<2x3x4x5xi32>>)
! CHECK:           %[[VAL_7:.*]] = fir.embox %[[VAL_6]]#0(%[[VAL_5]]) : (!fir.ref<!fir.array<2x3x4x5xi32>>, !fir.shape<4>) -> !fir.box<!fir.array<2x3x4x5xi32>>
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.box<!fir.array<2x3x4x5xi32>>) -> !fir.box<!fir.array<*:i32>>
! CHECK:           fir.call @_QPint_assumed_rank(%[[VAL_8]]) fastmath<contract> : (!fir.box<!fir.array<*:i32>>) -> ()

subroutine int_assumed_shape_to_assumed_rank(x)
  use ifaces, only : int_assumed_rank
  integer :: x(:, :)
  call int_assumed_rank(x)
end subroutine
! CHECK-LABEL:   func.func @_QPint_assumed_shape_to_assumed_rank(
! CHECK-SAME:                                                    %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFint_assumed_shape_to_assumed_rankEx"} : (!fir.box<!fir.array<?x?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?x?xi32>>)
! CHECK:           %[[VAL_2:.*]] = fir.convert %[[VAL_1]]#0 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<*:i32>>
! CHECK:           fir.call @_QPint_assumed_rank(%[[VAL_2]]) fastmath<contract> : (!fir.box<!fir.array<*:i32>>) -> ()

subroutine int_assumed_shape_to_assumed_rank_bindc(x)
  use ifaces, only : int_assumed_rank_bindc
  integer :: x(:, :)
  call int_assumed_rank_bindc(x)
end subroutine
! CHECK-LABEL:   func.func @_QPint_assumed_shape_to_assumed_rank_bindc(
! CHECK-SAME:                                                          %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFint_assumed_shape_to_assumed_rank_bindcEx"} : (!fir.box<!fir.array<?x?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?x?xi32>>)
! CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_3:.*]] = fir.shift %[[VAL_2]], %[[VAL_2]] : (index, index) -> !fir.shift<2>
! CHECK:           %[[VAL_4:.*]] = fir.rebox %[[VAL_1]]#0(%[[VAL_3]]) : (!fir.box<!fir.array<?x?xi32>>, !fir.shift<2>) -> !fir.box<!fir.array<?x?xi32>>
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<*:i32>>
! CHECK:           fir.call @int_assumed_rank_bindc(%[[VAL_5]]) proc_attrs<bind_c> fastmath<contract> : (!fir.box<!fir.array<*:i32>>) -> ()

subroutine int_allocatable_to_assumed_rank(x)
  use ifaces, only : int_assumed_rank
  integer, allocatable :: x(:, :)
  call int_assumed_rank(x)
end subroutine
! CHECK-LABEL:   func.func @_QPint_allocatable_to_assumed_rank(
! CHECK-SAME:                                                  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFint_allocatable_to_assumed_rankEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>)
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
! CHECK:           %[[VAL_3:.*]] = fir.rebox %[[VAL_2]] : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>) -> !fir.box<!fir.array<?x?xi32>>
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<*:i32>>
! CHECK:           fir.call @_QPint_assumed_rank(%[[VAL_4]]) fastmath<contract> : (!fir.box<!fir.array<*:i32>>) -> ()

subroutine int_allocatable_to_assumed_rank_opt(x)
  use ifaces, only : int_opt_assumed_rank
  integer, allocatable :: x(:, :)
  call int_opt_assumed_rank(x)
end subroutine
! CHECK-LABEL:   func.func @_QPint_allocatable_to_assumed_rank_opt(
! CHECK-SAME:                                                      %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFint_allocatable_to_assumed_rank_optEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>)
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_1]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
! CHECK:           %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>) -> !fir.heap<!fir.array<?x?xi32>>
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.heap<!fir.array<?x?xi32>>) -> i64
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_6:.*]] = arith.cmpi ne, %[[VAL_4]], %[[VAL_5]] : i64
! CHECK:           %[[VAL_7:.*]] = fir.if %[[VAL_6]] -> (!fir.box<!fir.array<?x?xi32>>) {
! CHECK:             %[[VAL_8:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
! CHECK:             %[[VAL_9:.*]] = fir.rebox %[[VAL_8]] : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>) -> !fir.box<!fir.array<?x?xi32>>
! CHECK:             fir.result %[[VAL_9]] : !fir.box<!fir.array<?x?xi32>>
! CHECK:           } else {
! CHECK:             %[[VAL_10:.*]] = fir.absent !fir.box<!fir.array<?x?xi32>>
! CHECK:             fir.result %[[VAL_10]] : !fir.box<!fir.array<?x?xi32>>
! CHECK:           }
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_7]] : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<*:i32>>
! CHECK:           fir.call @_QPint_opt_assumed_rank(%[[VAL_11]]) fastmath<contract> : (!fir.box<!fir.array<*:i32>>) -> ()

subroutine int_r2_assumed_size_to_assumed_rank(x)
  use ifaces, only : int_assumed_rank
  integer :: x(10, *)
  call int_assumed_rank(x)
end subroutine
! CHECK-LABEL:   func.func @_QPint_r2_assumed_size_to_assumed_rank(
! CHECK-SAME:                                                      %[[VAL_0:.*]]: !fir.ref<!fir.array<10x?xi32>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = arith.constant 10 : i64
! CHECK:           %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (i64) -> index
! CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_4:.*]] = arith.cmpi sgt, %[[VAL_2]], %[[VAL_3]] : index
! CHECK:           %[[VAL_5:.*]] = arith.select %[[VAL_4]], %[[VAL_2]], %[[VAL_3]] : index
! CHECK:           %[[VAL_6:.*]] = arith.constant -1 : index
! CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_5]], %[[VAL_6]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_7]]) dummy_scope %{{[0-9]+}} {uniq_name = "_QFint_r2_assumed_size_to_assumed_rankEx"} : (!fir.ref<!fir.array<10x?xi32>>, !fir.shape<2>, !fir.dscope) -> (!fir.box<!fir.array<10x?xi32>>, !fir.ref<!fir.array<10x?xi32>>)
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_8]]#0 : (!fir.box<!fir.array<10x?xi32>>) -> !fir.box<!fir.array<*:i32>>
! CHECK:           fir.call @_QPint_assumed_rank(%[[VAL_9]]) fastmath<contract> : (!fir.box<!fir.array<*:i32>>) -> ()
