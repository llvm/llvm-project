! Test passing of assumed-ranks that require creating a
! a new descriptor for the dummy argument (different lower bounds,
! attribute, or dynamic type)
! RUN: bbc -emit-hlfir -allow-assumed-rank -o - %s | FileCheck %s

subroutine test_alloc_to_nonalloc(x)
  real, allocatable ::  x(..)
  interface
    subroutine takes_assumed_rank(x)
      real :: x(..)
    end subroutine
  end interface
  call takes_assumed_rank(x)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_alloc_to_nonalloc(
! CHECK-SAME:                                         %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_alloc_to_nonallocEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>)
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>
! CHECK:           %[[VAL_4:.*]] = fir.rebox_assumed_rank %[[VAL_3]] lbs ones : (!fir.box<!fir.heap<!fir.array<*:f32>>>) -> !fir.box<!fir.array<*:f32>>
! CHECK:           fir.call @_QPtakes_assumed_rank(%[[VAL_4]]) fastmath<contract> : (!fir.box<!fir.array<*:f32>>) -> ()
! CHECK:           return
! CHECK:         }

subroutine test_to_bindc(x)
  real ::  x(..)
  interface
    subroutine bindc_func(x) bind(c)
      real :: x(..)
    end subroutine
  end interface
  call bindc_func(x)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_to_bindc(
! CHECK-SAME:                                %[[VAL_0:.*]]: !fir.box<!fir.array<*:f32>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {uniq_name = "_QFtest_to_bindcEx"} : (!fir.box<!fir.array<*:f32>>, !fir.dscope) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           %[[VAL_3:.*]] = fir.rebox_assumed_rank %[[VAL_2]]#0 lbs zeroes : (!fir.box<!fir.array<*:f32>>) -> !fir.box<!fir.array<*:f32>>
! CHECK:           fir.call @bindc_func(%[[VAL_3]]) fastmath<contract> {is_bind_c} : (!fir.box<!fir.array<*:f32>>) -> ()
! CHECK:           return
! CHECK:         }

subroutine test_poly_to_nonepoly(x)
  type t
    integer :: i
  end type
  class(t) ::  x(..)
  interface
    subroutine takes_assumed_rank_t(x)
      import :: t
      type(t) :: x(..)
    end subroutine
  end interface
  call takes_assumed_rank_t(x)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_poly_to_nonepoly(
! CHECK-SAME:                                        %[[VAL_0:.*]]: !fir.class<!fir.array<*:!fir.type<_QFtest_poly_to_nonepolyTt{i:i32}>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {uniq_name = "_QFtest_poly_to_nonepolyEx"} : (!fir.class<!fir.array<*:!fir.type<_QFtest_poly_to_nonepolyTt{i:i32}>>>, !fir.dscope) -> (!fir.class<!fir.array<*:!fir.type<_QFtest_poly_to_nonepolyTt{i:i32}>>>, !fir.class<!fir.array<*:!fir.type<_QFtest_poly_to_nonepolyTt{i:i32}>>>)
! CHECK:           %[[VAL_3:.*]] = fir.rebox_assumed_rank %[[VAL_2]]#0 lbs ones : (!fir.class<!fir.array<*:!fir.type<_QFtest_poly_to_nonepolyTt{i:i32}>>>) -> !fir.box<!fir.array<*:!fir.type<_QFtest_poly_to_nonepolyTt{i:i32}>>>
! CHECK:           fir.call @_QPtakes_assumed_rank_t(%[[VAL_3]]) fastmath<contract> : (!fir.box<!fir.array<*:!fir.type<_QFtest_poly_to_nonepolyTt{i:i32}>>>) -> ()
! CHECK:           return
! CHECK:         }
