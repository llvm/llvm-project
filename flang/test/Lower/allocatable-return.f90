! RUN: %flang_fc1 -emit-hlfir -I nowhere %s -o - | FileCheck %s

! Test allocatable return.
! Allocatable arrays must have default runtime lbounds after the return.

function test_alloc_return_scalar
  real, allocatable :: test_alloc_return_scalar
  allocate(test_alloc_return_scalar)
end function test_alloc_return_scalar
! CHECK-LABEL:   func.func @_QPtest_alloc_return_scalar() -> !fir.box<!fir.heap<f32>> {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<f32>> {bindc_name = "test_alloc_return_scalar", uniq_name = "_QFtest_alloc_return_scalarEtest_alloc_return_scalar"}
! CHECK:           %[[VAL_DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}
! CHECK:           %[[VAL:.*]] = fir.load %[[VAL_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:           return %[[VAL]] : !fir.box<!fir.heap<f32>>
! CHECK:         }

function test_alloc_return_array
  real, allocatable :: test_alloc_return_array(:)
  allocate(test_alloc_return_array(7:8))
end function test_alloc_return_array
! CHECK-LABEL:   func.func @_QPtest_alloc_return_array() -> !fir.box<!fir.heap<!fir.array<?xf32>>> {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "test_alloc_return_array", uniq_name = "_QFtest_alloc_return_arrayEtest_alloc_return_array"}
! CHECK:           %[[VAL_DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}
! CHECK:           %[[VAL:.*]] = fir.load %[[VAL_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:           %[[C1:.*]] = arith.constant 1 : index
! CHECK:           %[[SHIFT:.*]] = fir.shift %[[C1]] : (index) -> !fir.shift<1>
! CHECK:           %[[REBOX:.*]] = fir.rebox %[[VAL]](%[[SHIFT]]) : (!fir.box<!fir.heap<!fir.array<?xf32>>>, !fir.shift<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:           return %[[REBOX]] : !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:         }

function test_alloc_return_char_scalar
  character(3), allocatable :: test_alloc_return_char_scalar
  allocate(test_alloc_return_char_scalar)
end function test_alloc_return_char_scalar
! CHECK-LABEL:   func.func @_QPtest_alloc_return_char_scalar() -> !fir.box<!fir.heap<!fir.char<1,3>>> {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,3>>> {bindc_name = "test_alloc_return_char_scalar", uniq_name = "_QFtest_alloc_return_char_scalarEtest_alloc_return_char_scalar"}
! CHECK:           %[[LEN:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_DECL:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[LEN]] {{.*}}
! CHECK:           %[[VAL:.*]] = fir.load %[[VAL_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,3>>>>
! CHECK:           return %[[VAL]] : !fir.box<!fir.heap<!fir.char<1,3>>>
! CHECK:         }

function test_alloc_return_char_array
  character(3), allocatable :: test_alloc_return_char_array(:)
  allocate(test_alloc_return_char_array(7:8))
end function test_alloc_return_char_array
! CHECK-LABEL:   func.func @_QPtest_alloc_return_char_array() -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,3>>>> {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,3>>>> {bindc_name = "test_alloc_return_char_array", uniq_name = "_QFtest_alloc_return_char_arrayEtest_alloc_return_char_array"}
! CHECK:           %[[LEN:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_DECL:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[LEN]] {{.*}}
! CHECK:           %[[VAL:.*]] = fir.load %[[VAL_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,3>>>>>
! CHECK:           %[[C1:.*]] = arith.constant 1 : index
! CHECK:           %[[SHIFT:.*]] = fir.shift %[[C1]] : (index) -> !fir.shift<1>
! CHECK:           %[[REBOX:.*]] = fir.rebox %[[VAL]](%[[SHIFT]]) : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,3>>>>, !fir.shift<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,3>>>>
! CHECK:           return %[[REBOX]] : !fir.box<!fir.heap<!fir.array<?x!fir.char<1,3>>>>
! CHECK:         }

function test_alloc_return_poly_scalar
  type t
  end type t
  class(*), allocatable :: test_alloc_return_poly_scalar
  allocate(t :: test_alloc_return_poly_scalar)
end function test_alloc_return_poly_scalar
! CHECK-LABEL:   func.func @_QPtest_alloc_return_poly_scalar() -> !fir.class<!fir.heap<none>> {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.class<!fir.heap<none>> {bindc_name = "test_alloc_return_poly_scalar", uniq_name = "_QFtest_alloc_return_poly_scalarEtest_alloc_return_poly_scalar"}
! CHECK:           %[[VAL_DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}
! CHECK:           %[[VAL:.*]] = fir.load %[[VAL_DECL]]#0 : !fir.ref<!fir.class<!fir.heap<none>>>
! CHECK:           return %[[VAL]] : !fir.class<!fir.heap<none>>
! CHECK:         }

function test_alloc_return_poly_array
  type t
  end type t
  class(*), allocatable :: test_alloc_return_poly_array(:)
  allocate(t :: test_alloc_return_poly_array(7:8))
end function test_alloc_return_poly_array
! CHECK-LABEL:   func.func @_QPtest_alloc_return_poly_array() -> !fir.class<!fir.heap<!fir.array<?xnone>>> {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?xnone>>> {bindc_name = "test_alloc_return_poly_array", uniq_name = "_QFtest_alloc_return_poly_arrayEtest_alloc_return_poly_array"}
! CHECK:           %[[VAL_DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}
! CHECK:           %[[VAL:.*]] = fir.load %[[VAL_DECL]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>
! CHECK:           %[[C1:.*]] = arith.constant 1 : index
! CHECK:           %[[SHIFT:.*]] = fir.shift %[[C1]] : (index) -> !fir.shift<1>
! CHECK:           %[[REBOX:.*]] = fir.rebox %[[VAL]](%[[SHIFT]]) : (!fir.class<!fir.heap<!fir.array<?xnone>>>, !fir.shift<1>) -> !fir.class<!fir.heap<!fir.array<?xnone>>>
! CHECK:           return %[[REBOX]] : !fir.class<!fir.heap<!fir.array<?xnone>>>
! CHECK:         }
