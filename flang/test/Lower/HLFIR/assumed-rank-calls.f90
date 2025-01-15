! Test passing of assumed-ranks that require creating a
! a new descriptor for the dummy argument (different lower bounds,
! attribute, or dynamic type)
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

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
! CHECK:           fir.call @bindc_func(%[[VAL_3]]) proc_attrs<bind_c> fastmath<contract> : (!fir.box<!fir.array<*:f32>>) -> ()
! CHECK:           return
! CHECK:         }

subroutine test_target_to_pointer(x)
  real, target ::  x(..)
  interface
    subroutine takes_target_as_pointer(x)
      real, pointer, intent(in) :: x(..)
    end subroutine
  end interface
  call takes_target_as_pointer(x)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_target_to_pointer(
! CHECK-SAME:                                         %[[VAL_0:.*]]: !fir.box<!fir.array<*:f32>> {fir.bindc_name = "x", fir.target}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<*:f32>>>
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtest_target_to_pointerEx"} : (!fir.box<!fir.array<*:f32>>, !fir.dscope) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           %[[VAL_4:.*]] = fir.rebox_assumed_rank %[[VAL_3]]#0 lbs preserve : (!fir.box<!fir.array<*:f32>>) -> !fir.box<!fir.ptr<!fir.array<*:f32>>>
! CHECK:           fir.store %[[VAL_4]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>
! CHECK:           fir.call @_QPtakes_target_as_pointer(%[[VAL_1]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>) -> ()

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


subroutine test_copy_in_out(x)
  real ::  x(..)
  interface
    subroutine takes_contiguous(x)
      real, contiguous :: x(..)
    end subroutine
  end interface
  call takes_contiguous(x)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_copy_in_out(
! CHECK-SAME:                                   %[[VAL_0:.*]]: !fir.box<!fir.array<*:f32>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<*:f32>>>
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] {uniq_name = "_QFtest_copy_in_outEx"} : (!fir.box<!fir.array<*:f32>>, !fir.dscope) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.copy_in %[[VAL_3]]#0 to %[[VAL_1]] : (!fir.box<!fir.array<*:f32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>) -> (!fir.box<!fir.array<*:f32>>, i1)
! CHECK:           fir.call @_QPtakes_contiguous(%[[VAL_4]]#0) fastmath<contract> : (!fir.box<!fir.array<*:f32>>) -> ()
! CHECK:           hlfir.copy_out %[[VAL_1]], %[[VAL_4]]#1 to %[[VAL_3]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>, i1, !fir.box<!fir.array<*:f32>>) -> ()

subroutine test_copy_in_out_2(x)
  real ::  x(..)
  interface
    subroutine takes_contiguous_intentin(x)
      real, intent(in), contiguous :: x(..)
    end subroutine
  end interface
  call takes_contiguous_intentin(x)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_copy_in_out_2(
! CHECK-SAME:                                     %[[VAL_0:.*]]: !fir.box<!fir.array<*:f32>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<*:f32>>>
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] {uniq_name = "_QFtest_copy_in_out_2Ex"} : (!fir.box<!fir.array<*:f32>>, !fir.dscope) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.copy_in %[[VAL_3]]#0 to %[[VAL_1]] : (!fir.box<!fir.array<*:f32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>) -> (!fir.box<!fir.array<*:f32>>, i1)
! CHECK:           fir.call @_QPtakes_contiguous_intentin(%[[VAL_4]]#0) fastmath<contract> : (!fir.box<!fir.array<*:f32>>) -> ()
! CHECK:           hlfir.copy_out %[[VAL_1]], %[[VAL_4]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>, i1) -> ()
