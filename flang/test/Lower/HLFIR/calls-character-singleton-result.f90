! Test handling of intrinsics and BIND(C) functions returning CHARACTER(1).
! This is a special case because characters are always returned
! or handled in memory otherwise.

! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine scalar_char(c, i)
  character(1) :: c
  integer(8) :: i
  c = char(i)
end subroutine
! CHECK-LABEL: func.func @_QPscalar_char(
! CHECK:  %[[VAL_2:.*]] = fir.alloca !fir.char<1>
! CHECK:  %[[VAL_4:.*]] = fir.convert %{{.*}}#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1>>
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] typeparams %{{.*}}  {{.*}}Ec
! CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare %{{.*}}  {{.*}}Ei
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i64) -> i8
! CHECK:  %[[VAL_9:.*]] = fir.undefined !fir.char<1>
! CHECK:  %[[VAL_10:.*]] = fir.insert_value %[[VAL_9]], %[[VAL_8]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK:  fir.store %[[VAL_10]] to %[[VAL_2]] : !fir.ref<!fir.char<1>>
! CHECK:  %[[VAL_11:.*]] = arith.constant false
! CHECK:  %[[VAL_12:.*]] = hlfir.as_expr %[[VAL_2]] move %[[VAL_11]] : (!fir.ref<!fir.char<1>>, i1) -> !hlfir.expr<!fir.char<1>>
! CHECK:  hlfir.assign %[[VAL_12]] to %[[VAL_5]]#0 : !hlfir.expr<!fir.char<1>>, !fir.ref<!fir.char<1>>

subroutine scalar_bindc(c)
  character(1) :: c
  interface
    character(1) function bar() bind(c)
    end function
  end interface
  c = bar()
end subroutine
! CHECK-LABEL: func.func @_QPscalar_bindc(
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.char<1>
! CHECK:  %[[VAL_3:.*]] = fir.convert %{{.*}}#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1>>
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] typeparams %{{.*}}  {{.*}}Ec
! CHECK:  %[[VAL_5:.*]] = fir.call @bar() proc_attrs<bind_c> fastmath<contract> : () -> !fir.char<1>
! CHECK:  fir.store %[[VAL_5]] to %[[VAL_1]] : !fir.ref<!fir.char<1>>
! CHECK:  %[[VAL_6:.*]] = arith.constant false
! CHECK:  %[[VAL_7:.*]] = hlfir.as_expr %[[VAL_1]] move %[[VAL_6]] : (!fir.ref<!fir.char<1>>, i1) -> !hlfir.expr<!fir.char<1>>
! CHECK:  hlfir.assign %[[VAL_7]] to %[[VAL_4]]#0 : !hlfir.expr<!fir.char<1>>, !fir.ref<!fir.char<1>>

subroutine array_char(c, i)
  character(1) :: c(100)
  integer(8) :: i(100)
  c = char(i)
end subroutine
! CHECK-LABEL: func.func @_QParray_char(
! CHECK:  %[[VAL_2:.*]] = fir.alloca !fir.char<1>
! CHECK:  %[[VAL_13:.*]] = hlfir.elemental %{{.*}} typeparams %{{.*}} : (!fir.shape<1>, index) -> !hlfir.expr<100x!fir.char<1>> {
! CHECK:  ^bb0(%[[VAL_14:.*]]: index):
! CHECK:    %[[VAL_19:.*]] = fir.insert_value {{.*}} -> !fir.char<1>
! CHECK:    fir.store %[[VAL_19]] to %[[VAL_2]] : !fir.ref<!fir.char<1>>
! CHECK:    %[[VAL_20:.*]] = arith.constant false
! CHECK:    %[[VAL_21:.*]] = hlfir.as_expr %[[VAL_2]] move %[[VAL_20]] : (!fir.ref<!fir.char<1>>, i1) -> !hlfir.expr<!fir.char<1>>
! CHECK:    hlfir.yield_element %[[VAL_21]] : !hlfir.expr<!fir.char<1>>
! CHECK:  }
! CHECK:  hlfir.assign %[[VAL_13]] to %{{.*}} : !hlfir.expr<100x!fir.char<1>>, !fir.ref<!fir.array<100x!fir.char<1>>>
