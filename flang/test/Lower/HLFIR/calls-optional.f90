! Test lowering of user calls involving passing an actual argument
! that is syntactically present, but may be absent at runtime (is
! an optional or a pointer/allocatable).
!
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine optional_copy_in_out(x)
  interface
    subroutine takes_optional_explicit(x)
      real, optional :: x(*)
    end subroutine
  end interface
  real, optional :: x(:)
  call  takes_optional_explicit(x)
end subroutine
! CHECK-LABEL: func.func @_QPoptional_copy_in_out(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFoptional_copy_in_outEx"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:  %[[VAL_2:.*]] = fir.is_present %[[VAL_1]]#0 : (!fir.box<!fir.array<?xf32>>) -> i1
! CHECK:  %[[VAL_3:.*]]:4 = fir.if %[[VAL_2]] -> (!fir.ref<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>, i1, !fir.box<!fir.array<?xf32>>) {
! CHECK:    %[[VAL_4:.*]]:2 = hlfir.copy_in %[[VAL_1]]#0 : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, i1)
! CHECK:    %[[VAL_5:.*]] = fir.box_addr %[[VAL_4]]#0 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:    fir.result %[[VAL_5]], %[[VAL_4]]#0, %[[VAL_4]]#1, %[[VAL_1]]#0 : !fir.ref<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>, i1, !fir.box<!fir.array<?xf32>>
! CHECK:  } else {
! CHECK:    %[[VAL_6:.*]] = fir.absent !fir.ref<!fir.array<?xf32>>
! CHECK:    %[[VAL_7:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
! CHECK:    %[[VAL_8:.*]] = arith.constant false
! CHECK:    %[[VAL_9:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
! CHECK:    fir.result %[[VAL_6]], %[[VAL_7]], %[[VAL_8]], %[[VAL_9]] : !fir.ref<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>, i1, !fir.box<!fir.array<?xf32>>
! CHECK:  }
! CHECK:  fir.call @_QPtakes_optional_explicit(%[[VAL_3]]#0) {{.*}} : (!fir.ref<!fir.array<?xf32>>) -> ()
! CHECK:  hlfir.copy_out %[[VAL_3]]#1, %[[VAL_3]]#2 to %[[VAL_3]]#3 : (!fir.box<!fir.array<?xf32>>, i1, !fir.box<!fir.array<?xf32>>) -> ()

subroutine optional_value_copy(x)
  interface
    subroutine takes_optional_explicit_value(x)
      real, value, optional :: x(100)
    end subroutine
  end interface
  real, optional :: x(100)
  call  takes_optional_explicit_value(x)
end subroutine
! CHECK-LABEL: func.func @_QPoptional_value_copy(
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]](%[[VAL_2:[a-z0-9]*]]) {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFoptional_value_copyEx"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
! CHECK:  %[[VAL_4:.*]] = fir.is_present %[[VAL_3]]#0 : (!fir.ref<!fir.array<100xf32>>) -> i1
! CHECK:  %[[VAL_5:.*]]:3 = fir.if %[[VAL_4]] -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>, i1) {
! CHECK:    %[[VAL_6:.*]] = hlfir.as_expr %[[VAL_3]]#0 : (!fir.ref<!fir.array<100xf32>>) -> !hlfir.expr<100xf32>
! CHECK:    %[[VAL_7:.*]]:3 = hlfir.associate %[[VAL_6]](%[[VAL_2]]) {adapt.valuebyref} : (!hlfir.expr<100xf32>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>, i1)
! CHECK:    fir.result %[[VAL_7]]#1, %[[VAL_7]]#1, %[[VAL_7]]#2 : !fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>, i1
! CHECK:  } else {
! CHECK:    %[[VAL_8:.*]] = fir.absent !fir.ref<!fir.array<100xf32>>
! CHECK:    %[[VAL_9:.*]] = fir.absent !fir.ref<!fir.array<100xf32>>
! CHECK:    %[[VAL_10:.*]] = arith.constant false
! CHECK:    fir.result %[[VAL_8]], %[[VAL_9]], %[[VAL_10]] : !fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>, i1
! CHECK:  }
! CHECK:  fir.call @_QPtakes_optional_explicit_value(%[[VAL_5]]#0) {{.*}} : (!fir.ref<!fir.array<100xf32>>) -> ()
! CHECK:  hlfir.end_associate %[[VAL_5]]#1, %[[VAL_5]]#2 : !fir.ref<!fir.array<100xf32>>, i1

subroutine elem_pointer_to_optional(x, y)
  interface
    elemental subroutine elem_takes_two_optional(x, y)
      real, optional, intent(in) :: y, x
    end subroutine
  end interface
  real :: x(:)
  real, pointer :: y(:)
  call elem_takes_two_optional(x, y)
end subroutine
! CHECK-LABEL: func.func @_QPelem_pointer_to_optional(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]] {uniq_name = "_QFelem_pointer_to_optionalEx"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1:[a-z0-9]*]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFelem_pointer_to_optionalEy"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_3]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_5:.*]] = fir.box_addr %[[VAL_4]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (!fir.ptr<!fir.array<?xf32>>) -> i64
! CHECK:  %[[VAL_7:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_8:.*]] = arith.cmpi ne, %[[VAL_6]], %[[VAL_7]] : i64
! CHECK:  %[[VAL_9:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_10:.*]]:3 = fir.box_dims %[[VAL_2]]#0, %[[VAL_9]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:  %[[VAL_11:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_12:.*]] = arith.constant 1 : index
! CHECK:  fir.do_loop %[[VAL_13:.*]] = %[[VAL_12]] to %[[VAL_10]]#1 step %[[VAL_12]] unordered {
! CHECK:    %[[VAL_14:.*]] = hlfir.designate %[[VAL_2]]#0 (%[[VAL_13]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! CHECK:    %[[VAL_15:.*]] = fir.if %[[VAL_8]] -> (!fir.ref<f32>) {
! CHECK:      %[[VAL_16:.*]] = arith.constant 0 : index
! CHECK:      %[[VAL_17:.*]]:3 = fir.box_dims %[[VAL_11]], %[[VAL_16]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:      %[[VAL_18:.*]] = arith.constant 1 : index
! CHECK:      %[[VAL_19:.*]] = arith.subi %[[VAL_17]]#0, %[[VAL_18]] : index
! CHECK:      %[[VAL_20:.*]] = arith.addi %[[VAL_13]], %[[VAL_19]] : index
! CHECK:      %[[VAL_21:.*]] = hlfir.designate %[[VAL_11]] (%[[VAL_20]])  : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> !fir.ref<f32>
! CHECK:      fir.result %[[VAL_21]] : !fir.ref<f32>
! CHECK:    } else {
! CHECK:      %[[VAL_22:.*]] = fir.absent !fir.ref<f32>
! CHECK:      fir.result %[[VAL_22]] : !fir.ref<f32>
! CHECK:    }
! CHECK:    fir.call @_QPelem_takes_two_optional(%[[VAL_14]], %[[VAL_15]]) {{.*}} : (!fir.ref<f32>, !fir.ref<f32>) -> ()
! CHECK:  }

subroutine optional_cannot_be_absent_optional(x)
  interface
    elemental subroutine elem_takes_one_optional(x)
      real, optional, intent(in) :: x
    end subroutine
  end interface
  real, optional :: x(:)
  ! If all array arguments in an call are optional, they must be all present.
  call elem_takes_one_optional(x)
end subroutine
! CHECK-LABEL: func.func @_QPoptional_cannot_be_absent_optional(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFoptional_cannot_be_absent_optionalEx"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:  %[[VAL_2:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_3:.*]]:3 = fir.box_dims %[[VAL_1]]#0, %[[VAL_2]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:  %[[VAL_4:.*]] = arith.constant 1 : index
! CHECK:  fir.do_loop %[[VAL_5:.*]] = %[[VAL_4]] to %[[VAL_3]]#1 step %[[VAL_4]] unordered {
! CHECK:    %[[VAL_6:.*]] = hlfir.designate %[[VAL_1]]#0 (%[[VAL_5]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! CHECK:    fir.call @_QPelem_takes_one_optional(%[[VAL_6]]) {{.*}} : (!fir.ref<f32>) -> ()
! CHECK:  }

subroutine optional_elem_poly(x, y)
  interface
    elemental subroutine elem_optional_poly(x, y)
      class(*), optional, intent(in) :: x, y
    end subroutine
  end interface
  real :: x(:)
  real, optional :: y(:)
  call elem_optional_poly(x, y)
end subroutine
! CHECK-LABEL: func.func @_QPoptional_elem_poly(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]] {uniq_name = "_QFoptional_elem_polyEx"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1:[a-z0-9]*]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFoptional_elem_polyEy"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:  %[[VAL_4:.*]] = fir.is_present %[[VAL_3]]#0 : (!fir.box<!fir.array<?xf32>>) -> i1
! CHECK:  %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_2]]#0, %[[VAL_5]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:  %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:  fir.do_loop %[[VAL_8:.*]] = %[[VAL_7]] to %[[VAL_6]]#1 step %[[VAL_7]] unordered {
! CHECK:    %[[VAL_9:.*]] = hlfir.designate %[[VAL_2]]#0 (%[[VAL_8]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! CHECK:    %[[VAL_10:.*]] = fir.embox %[[VAL_9]] : (!fir.ref<f32>) -> !fir.box<f32>
! CHECK:    %[[VAL_11:.*]] = fir.rebox %[[VAL_10]] : (!fir.box<f32>) -> !fir.class<none>
! CHECK:    %[[VAL_12:.*]] = fir.if %[[VAL_4]] -> (!fir.class<none>) {
! CHECK:      %[[VAL_13:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_8]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! CHECK:      %[[VAL_14:.*]] = fir.embox %[[VAL_13]] : (!fir.ref<f32>) -> !fir.box<f32>
! CHECK:      %[[VAL_15:.*]] = fir.rebox %[[VAL_14]] : (!fir.box<f32>) -> !fir.class<none>
! CHECK:      fir.result %[[VAL_15]] : !fir.class<none>
! CHECK:    } else {
! CHECK:      %[[VAL_16:.*]] = fir.absent !fir.class<none>
! CHECK:      fir.result %[[VAL_16]] : !fir.class<none>
! CHECK:    }
! CHECK:    fir.call @_QPelem_optional_poly(%[[VAL_11]], %[[VAL_12]]) {{.*}} : (!fir.class<none>, !fir.class<none>) -> ()
! CHECK:  }

subroutine test_passing_null()
  interface
    subroutine takes_optional_assumed(x)
      real, optional :: x(:)
    end subroutine
  end interface
  call takes_optional_assumed(null())
 ! NULL(MOLD) lowering is a TODO in HLFIR.
 ! call takes_optional_assumed(null(p))
end subroutine
! CHECK-LABEL: func.func @_QPtest_passing_null() {
! CHECK:  %[[VAL_0:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
! CHECK:  fir.call @_QPtakes_optional_assumed(%[[VAL_0]]) {{.*}} : (!fir.box<!fir.array<?xf32>>) -> ()
