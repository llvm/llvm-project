! RUN: bbc -emit-fir -hlfir=false -I nowhere %s -o - | FileCheck %s

! Test allocatable return.
! Allocatable arrays must have default runtime lbounds after the return.

function test_alloc_return_scalar
  real, allocatable :: test_alloc_return_scalar
  allocate(test_alloc_return_scalar)
end function test_alloc_return_scalar
! CHECK-LABEL:   func.func @_QPtest_alloc_return_scalar() -> !fir.box<!fir.heap<f32>> {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<f32>> {bindc_name = "test_alloc_return_scalar", uniq_name = "_QFtest_alloc_return_scalarEtest_alloc_return_scalar"}
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:           return %[[VAL_5]] : !fir.box<!fir.heap<f32>>
! CHECK:         }

function test_alloc_return_array
  real, allocatable :: test_alloc_return_array(:)
  allocate(test_alloc_return_array(7:8))
end function test_alloc_return_array
! CHECK-LABEL:   func.func @_QPtest_alloc_return_array() -> !fir.box<!fir.heap<!fir.array<?xf32>>> {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "test_alloc_return_array", uniq_name = "_QFtest_alloc_return_arrayEtest_alloc_return_array"}
! CHECK:           %[[VAL_18:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:           %[[VAL_19:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_20:.*]] = fir.shift %[[VAL_19]] : (index) -> !fir.shift<1>
! CHECK:           %[[VAL_21:.*]] = fir.rebox %[[VAL_18]](%[[VAL_20]]) : (!fir.box<!fir.heap<!fir.array<?xf32>>>, !fir.shift<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:           return %[[VAL_21]] : !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:         }

function test_alloc_return_char_scalar
  character(3), allocatable :: test_alloc_return_char_scalar
  allocate(test_alloc_return_char_scalar)
end function test_alloc_return_char_scalar
! CHECK-LABEL:   func.func @_QPtest_alloc_return_char_scalar() -> !fir.box<!fir.heap<!fir.char<1,3>>> {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,3>>> {bindc_name = "test_alloc_return_char_scalar", uniq_name = "_QFtest_alloc_return_char_scalarEtest_alloc_return_char_scalar"}
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,3>>>>
! CHECK:           return %[[VAL_5]] : !fir.box<!fir.heap<!fir.char<1,3>>>
! CHECK:         }

function test_alloc_return_char_array
  character(3), allocatable :: test_alloc_return_char_array(:)
  allocate(test_alloc_return_char_array(7:8))
end function test_alloc_return_char_array
! CHECK-LABEL:   func.func @_QPtest_alloc_return_char_array() -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,3>>>> {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,3>>>> {bindc_name = "test_alloc_return_char_array", uniq_name = "_QFtest_alloc_return_char_arrayEtest_alloc_return_char_array"}
! CHECK:           %[[VAL_18:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,3>>>>>
! CHECK:           %[[VAL_19:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_20:.*]] = fir.shift %[[VAL_19]] : (index) -> !fir.shift<1>
! CHECK:           %[[VAL_21:.*]] = fir.rebox %[[VAL_18]](%[[VAL_20]]) : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,3>>>>, !fir.shift<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,3>>>>
! CHECK:           return %[[VAL_21]] : !fir.box<!fir.heap<!fir.array<?x!fir.char<1,3>>>>
! CHECK:         }

function test_alloc_return_poly_scalar
  type t
  end type t
  class(*), allocatable :: test_alloc_return_poly_scalar
  allocate(t :: test_alloc_return_poly_scalar)
end function test_alloc_return_poly_scalar
! CHECK-LABEL:   func.func @_QPtest_alloc_return_poly_scalar() -> !fir.class<!fir.heap<none>> {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.class<!fir.heap<none>> {bindc_name = "test_alloc_return_poly_scalar", uniq_name = "_QFtest_alloc_return_poly_scalarEtest_alloc_return_poly_scalar"}
! CHECK:           %[[VAL_16:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.class<!fir.heap<none>>>
! CHECK:           return %[[VAL_16]] : !fir.class<!fir.heap<none>>
! CHECK:         }

function test_alloc_return_poly_array
  type t
  end type t
  class(*), allocatable :: test_alloc_return_poly_array(:)
  allocate(t :: test_alloc_return_poly_array(7:8))
end function test_alloc_return_poly_array
! CHECK-LABEL:   func.func @_QPtest_alloc_return_poly_array() -> !fir.class<!fir.heap<!fir.array<?xnone>>> {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?xnone>>> {bindc_name = "test_alloc_return_poly_array", uniq_name = "_QFtest_alloc_return_poly_arrayEtest_alloc_return_poly_array"}
! CHECK:           %[[VAL_25:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>
! CHECK:           %[[VAL_26:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_27:.*]] = fir.shift %[[VAL_26]] : (index) -> !fir.shift<1>
! CHECK:           %[[VAL_28:.*]] = fir.rebox %[[VAL_25]](%[[VAL_27]]) : (!fir.class<!fir.heap<!fir.array<?xnone>>>, !fir.shift<1>) -> !fir.class<!fir.heap<!fir.array<?xnone>>>
! CHECK:           return %[[VAL_28]] : !fir.class<!fir.heap<!fir.array<?xnone>>>
! CHECK:         }
