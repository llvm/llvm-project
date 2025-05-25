! Test lowering of calls to interface with pointer or allocatable
! assumed rank dummy arguments.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

module ifaces_ptr_alloc
  interface
    subroutine alloc_assumed_rank(y)
      real, allocatable :: y(..)
    end subroutine
    subroutine pointer_assumed_rank(y)
      real, optional, pointer :: y(..)
    end subroutine
    subroutine pointer_assumed_rank2(y)
      real, intent(in), pointer :: y(..)
    end subroutine
  end interface
end module

subroutine scalar_alloc_to_assumed_rank(x)
  use ifaces_ptr_alloc, only : alloc_assumed_rank
  real, allocatable :: x
  call alloc_assumed_rank(x)
end subroutine
! CHECK-LABEL:   func.func @_QPscalar_alloc_to_assumed_rank(
! CHECK-SAME:                                               %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<f32>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFscalar_alloc_to_assumed_rankEx"} : (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)
! CHECK:           %[[VAL_2:.*]] = fir.convert %[[VAL_1]]#0 : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>
! CHECK:           fir.call @_QPalloc_assumed_rank(%[[VAL_2]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>) -> ()

subroutine r2_alloc_to_assumed_rank(x)
  use ifaces_ptr_alloc, only : alloc_assumed_rank
  real, allocatable :: x(:, :)
  call alloc_assumed_rank(x)
end subroutine
! CHECK-LABEL:   func.func @_QPr2_alloc_to_assumed_rank(
! CHECK-SAME:                                           %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFr2_alloc_to_assumed_rankEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>)
! CHECK:           %[[VAL_2:.*]] = fir.convert %[[VAL_1]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>
! CHECK:           fir.call @_QPalloc_assumed_rank(%[[VAL_2]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>) -> ()

subroutine scalar_pointer_to_assumed_rank(x)
  use ifaces_ptr_alloc, only : pointer_assumed_rank
  real, pointer :: x
  call pointer_assumed_rank(x)
end subroutine
! CHECK-LABEL:   func.func @_QPscalar_pointer_to_assumed_rank(
! CHECK-SAME:                                                 %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<f32>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFscalar_pointer_to_assumed_rankEx"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.ptr<f32>>>)
! CHECK:           %[[VAL_2:.*]] = fir.convert %[[VAL_1]]#0 : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>
! CHECK:           fir.call @_QPpointer_assumed_rank(%[[VAL_2]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>) -> ()

subroutine r2_pointer_to_assumed_rank(x)
  use ifaces_ptr_alloc, only : pointer_assumed_rank
  real, pointer :: x(:, :)
  call pointer_assumed_rank(x)
end subroutine
! CHECK-LABEL:   func.func @_QPr2_pointer_to_assumed_rank(
! CHECK-SAME:                                             %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFr2_pointer_to_assumed_rankEx"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>)
! CHECK:           %[[VAL_2:.*]] = fir.convert %[[VAL_1]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>
! CHECK:           fir.call @_QPpointer_assumed_rank(%[[VAL_2]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>) -> ()

subroutine r2_target_to_pointer_assumed_rank(x)
  use ifaces_ptr_alloc, only : pointer_assumed_rank2
  real, target :: x(:, :)
  call pointer_assumed_rank2(x)
end subroutine
! CHECK-LABEL:   func.func @_QPr2_target_to_pointer_assumed_rank(
! CHECK-SAME:                                                    %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x", fir.target}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFr2_target_to_pointer_assumed_rankEx"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
! CHECK:           %[[VAL_3:.*]] = fir.rebox %[[VAL_2]]#1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
! CHECK:           fir.store %[[VAL_3]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>
! CHECK:           fir.call @_QPpointer_assumed_rank2(%[[VAL_4]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>) -> ()
