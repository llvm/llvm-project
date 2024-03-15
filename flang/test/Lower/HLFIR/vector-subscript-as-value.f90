! Test lowering of vector subscript designators outside of the
! assignment left-and side and input IO context.
! RUN: bbc -emit-hlfir -o - -I nw %s --polymorphic-type 2>&1 | FileCheck %s

subroutine foo(x, y)
  integer :: x(100)
  integer(8) :: y(20)
  call bar(x(y))
end subroutine
! CHECK-LABEL:   func.func @_QPfoo(
! CHECK:  %[[VAL_2:.*]] = arith.constant 100 : index
! CHECK:  %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]](%[[VAL_3:[a-z0-9]*]])  {{.*}}Ex
! CHECK:  %[[VAL_5:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_1:[a-z0-9]*]](%[[VAL_6:[a-z0-9]*]])  {{.*}}Ey
! CHECK:  %[[VAL_8:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_9:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_10:.*]] = hlfir.elemental %[[VAL_9]] unordered : (!fir.shape<1>) -> !hlfir.expr<20xi32> {
! CHECK:  ^bb0(%[[VAL_11:.*]]: index):
! CHECK:    %[[VAL_12:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_11]])  : (!fir.ref<!fir.array<20xi64>>, index) -> !fir.ref<i64>
! CHECK:    %[[VAL_13:.*]] = fir.load %[[VAL_12]] : !fir.ref<i64>
! CHECK:    %[[VAL_14:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_13]])  : (!fir.ref<!fir.array<100xi32>>, i64) -> !fir.ref<i32>
! CHECK:    %[[VAL_15:.*]] = fir.load %[[VAL_14]] : !fir.ref<i32>
! CHECK:    hlfir.yield_element %[[VAL_15]] : i32
! CHECK:  }
! CHECK:  %[[VAL_16:.*]]:3 = hlfir.associate %[[VAL_17:.*]](%[[VAL_9]]) {adapt.valuebyref} : (!hlfir.expr<20xi32>, !fir.shape<1>) -> (!fir.ref<!fir.array<20xi32>>, !fir.ref<!fir.array<20xi32>>, i1)
! CHECK:  fir.call @_QPbar(%[[VAL_16]]#1) fastmath<contract> : (!fir.ref<!fir.array<20xi32>>) -> ()
! CHECK:  hlfir.end_associate %[[VAL_16]]#1, %[[VAL_16]]#2 : !fir.ref<!fir.array<20xi32>>, i1
! CHECK:  hlfir.destroy %[[VAL_17]] : !hlfir.expr<20xi32>

subroutine foo2(x, y)
  integer :: x(10, 30, 100)
  integer(8) :: y(20)
  call bar2(x(1:8:2, 5, y))
end subroutine
! CHECK-LABEL:   func.func @_QPfoo2(
! CHECK:  %[[VAL_2:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_3:.*]] = arith.constant 30 : index
! CHECK:  %[[VAL_4:.*]] = arith.constant 100 : index
! CHECK:  %[[VAL_5:.*]] = fir.shape %[[VAL_2]], %[[VAL_3]], %[[VAL_4]] : (index, index, index) -> !fir.shape<3>
! CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]](%[[VAL_5:[a-z0-9]*]])  {{.*}}Ex
! CHECK:  %[[VAL_7:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_8:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_1:[a-z0-9]*]](%[[VAL_8:[a-z0-9]*]])  {{.*}}Ey
! CHECK:  %[[VAL_10:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_11:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_12:.*]] = arith.constant 4 : index
! CHECK:  %[[VAL_13:.*]] = arith.constant 5 : index
! CHECK:  %[[VAL_14:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_15:.*]] = fir.shape %[[VAL_12]], %[[VAL_14]] : (index, index) -> !fir.shape<2>
! CHECK:  %[[VAL_16:.*]] = hlfir.elemental %[[VAL_15]] unordered : (!fir.shape<2>) -> !hlfir.expr<4x20xi32> {
! CHECK:  ^bb0(%[[VAL_17:.*]]: index, %[[VAL_18:.*]]: index):
! CHECK:    %[[VAL_19:.*]] = arith.constant 1 : index
! CHECK:    %[[VAL_20:.*]] = arith.subi %[[VAL_17]], %[[VAL_19]] : index
! CHECK:    %[[VAL_21:.*]] = arith.muli %[[VAL_20]], %[[VAL_11]] : index
! CHECK:    %[[VAL_22:.*]] = arith.addi %[[VAL_10]], %[[VAL_21]] : index
! CHECK:    %[[VAL_23:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_18]])  : (!fir.ref<!fir.array<20xi64>>, index) -> !fir.ref<i64>
! CHECK:    %[[VAL_24:.*]] = fir.load %[[VAL_23]] : !fir.ref<i64>
! CHECK:    %[[VAL_25:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_22]], %[[VAL_13]], %[[VAL_24]])  : (!fir.ref<!fir.array<10x30x100xi32>>, index, index, i64) -> !fir.ref<i32>
! CHECK:    %[[VAL_26:.*]] = fir.load %[[VAL_25]] : !fir.ref<i32>
! CHECK:    hlfir.yield_element %[[VAL_26]] : i32
! CHECK:  }

subroutine foo3(x, y)
  integer, pointer :: x(:, :, :)
  integer(8) :: y(20)
  call bar2(x(1:8:2, 5, y))
end subroutine
! CHECK-LABEL:   func.func @_QPfoo3(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFfoo3Ex"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x?xi32>>>>)
! CHECK:  %[[VAL_3:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_1:[a-z0-9]*]](%[[VAL_4:[a-z0-9]*]])  {{.*}}Ey
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x?xi32>>>>
! CHECK:  %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_8:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_9:.*]] = arith.constant 4 : index
! CHECK:  %[[VAL_10:.*]] = arith.constant 5 : index
! CHECK:  %[[VAL_11:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_12:.*]] = fir.shape %[[VAL_9]], %[[VAL_11]] : (index, index) -> !fir.shape<2>
! CHECK:  %[[VAL_13:.*]] = hlfir.elemental %[[VAL_12]] unordered : (!fir.shape<2>) -> !hlfir.expr<4x20xi32> {
! CHECK:  ^bb0(%[[VAL_14:.*]]: index, %[[VAL_15:.*]]: index):
! CHECK:    %[[VAL_16:.*]] = arith.constant 1 : index
! CHECK:    %[[VAL_17:.*]] = arith.subi %[[VAL_14]], %[[VAL_16]] : index
! CHECK:    %[[VAL_18:.*]] = arith.muli %[[VAL_17]], %[[VAL_8]] : index
! CHECK:    %[[VAL_19:.*]] = arith.addi %[[VAL_7]], %[[VAL_18]] : index
! CHECK:    %[[VAL_20:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_15]])  : (!fir.ref<!fir.array<20xi64>>, index) -> !fir.ref<i64>
! CHECK:    %[[VAL_21:.*]] = fir.load %[[VAL_20]] : !fir.ref<i64>
! CHECK:    %[[VAL_22:.*]] = hlfir.designate %[[VAL_6]] (%[[VAL_19]], %[[VAL_10]], %[[VAL_21]])  : (!fir.box<!fir.ptr<!fir.array<?x?x?xi32>>>, index, index, i64) -> !fir.ref<i32>
! CHECK:    %[[VAL_23:.*]] = fir.load %[[VAL_22]] : !fir.ref<i32>
! CHECK:    hlfir.yield_element %[[VAL_23]] : i32
! CHECK:  }

subroutine foo4(at1, vector, i, j, k, l, step)
  type t0
    complex :: x(10, 20)
  end type
  type t1
    type(t0) :: at0(30, 40, 50)
  end type
  type(t1) :: at1(:)
  integer(8) :: vector(:), step, i, j, k, l
  call bar3(at1(i)%at0(1:8:step, j, vector)%x(k, l)%im)
end subroutine
! CHECK-LABEL:   func.func @_QPfoo4(
! CHECK:  %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Eat1
! CHECK:  %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_2:[a-z0-9]*]]  {{.*}}Ei
! CHECK:  %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_3:[a-z0-9]*]]  {{.*}}Ej
! CHECK:  %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_4:[a-z0-9]*]]  {{.*}}Ek
! CHECK:  %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_5:[a-z0-9]*]]  {{.*}}El
! CHECK:  %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_6:[a-z0-9]*]]  {{.*}}Estep
! CHECK:  %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_1:[a-z0-9]*]]  {{.*}}Evector
! CHECK:  %[[VAL_14:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_15:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_14]])  : (!fir.box<!fir.array<?x!fir.type<_QFfoo4Tt1{at0:!fir.array<30x40x50x!fir.type<_QFfoo4Tt0{x:!fir.array<10x20x!fir.complex<4>>}>>}>>>, i64) -> !fir.ref<!fir.type<_QFfoo4Tt1{at0:!fir.array<30x40x50x!fir.type<_QFfoo4Tt0{x:!fir.array<10x20x!fir.complex<4>>}>>}>>
! CHECK:  %[[VAL_16:.*]] = arith.constant 30 : index
! CHECK:  %[[VAL_17:.*]] = arith.constant 40 : index
! CHECK:  %[[VAL_18:.*]] = arith.constant 50 : index
! CHECK:  %[[VAL_19:.*]] = fir.shape %[[VAL_16]], %[[VAL_17]], %[[VAL_18]] : (index, index, index) -> !fir.shape<3>
! CHECK:  %[[VAL_20:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_21:.*]] = arith.constant 8 : index
! CHECK:  %[[VAL_22:.*]] = fir.load %[[VAL_12]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i64) -> index
! CHECK:  %[[VAL_24:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_25:.*]] = arith.subi %[[VAL_21]], %[[VAL_20]] : index
! CHECK:  %[[VAL_26:.*]] = arith.addi %[[VAL_25]], %[[VAL_23]] : index
! CHECK:  %[[VAL_27:.*]] = arith.divsi %[[VAL_26]], %[[VAL_23]] : index
! CHECK:  %[[VAL_28:.*]] = arith.cmpi sgt, %[[VAL_27]], %[[VAL_24]] : index
! CHECK:  %[[VAL_29:.*]] = arith.select %[[VAL_28]], %[[VAL_27]], %[[VAL_24]] : index
! CHECK:  %[[VAL_30:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_31:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_32:.*]]:3 = fir.box_dims %[[VAL_13]]#0, %[[VAL_31]] : (!fir.box<!fir.array<?xi64>>, index) -> (index, index, index)
! CHECK:  %[[VAL_33:.*]] = fir.shape %[[VAL_29]], %[[VAL_32]]#1 : (index, index) -> !fir.shape<2>
! CHECK:  %[[VAL_34:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_35:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_36:.*]] = fir.shape %[[VAL_34]], %[[VAL_35]] : (index, index) -> !fir.shape<2>
! CHECK:  %[[VAL_37:.*]] = fir.load %[[VAL_10]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_38:.*]] = fir.load %[[VAL_11]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_39:.*]] = hlfir.elemental %[[VAL_33]] unordered : (!fir.shape<2>) -> !hlfir.expr<?x?xf32> {
! CHECK:  ^bb0(%[[VAL_40:.*]]: index, %[[VAL_41:.*]]: index):
! CHECK:    %[[VAL_42:.*]] = arith.constant 1 : index
! CHECK:    %[[VAL_43:.*]] = arith.subi %[[VAL_40]], %[[VAL_42]] : index
! CHECK:    %[[VAL_44:.*]] = arith.muli %[[VAL_43]], %[[VAL_23]] : index
! CHECK:    %[[VAL_45:.*]] = arith.addi %[[VAL_20]], %[[VAL_44]] : index
! CHECK:    %[[VAL_46:.*]] = hlfir.designate %[[VAL_13]]#0 (%[[VAL_41]])  : (!fir.box<!fir.array<?xi64>>, index) -> !fir.ref<i64>
! CHECK:    %[[VAL_47:.*]] = fir.load %[[VAL_46]] : !fir.ref<i64>
! CHECK:    %[[VAL_48:.*]] = hlfir.designate %[[VAL_15]]{"at0"} <%[[VAL_19]]> (%[[VAL_45]], %[[VAL_30]], %[[VAL_47]])  : (!fir.ref<!fir.type<_QFfoo4Tt1{at0:!fir.array<30x40x50x!fir.type<_QFfoo4Tt0{x:!fir.array<10x20x!fir.complex<4>>}>>}>>, !fir.shape<3>, index, i64, i64) -> !fir.ref<!fir.type<_QFfoo4Tt0{x:!fir.array<10x20x!fir.complex<4>>}>>
! CHECK:    %[[VAL_49:.*]] = hlfir.designate %[[VAL_48]]{"x"} <%[[VAL_36]]> (%[[VAL_37]], %[[VAL_38]]) imag : (!fir.ref<!fir.type<_QFfoo4Tt0{x:!fir.array<10x20x!fir.complex<4>>}>>, !fir.shape<2>, i64, i64) -> !fir.ref<f32>
! CHECK:    %[[VAL_50:.*]] = fir.load %[[VAL_49]] : !fir.ref<f32>
! CHECK:    hlfir.yield_element %[[VAL_50]] : f32
! CHECK:  }

subroutine substring(c, vector, i, j)
  character(*) :: c(:)
  integer(8) :: vector(:), step, i, j
  call bar4(c(vector)(i:j))
end subroutine
! CHECK-LABEL:   func.func @_QPsubstring(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ec
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_2:[a-z0-9]*]]  {{.*}}Ei
! CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_3:[a-z0-9]*]]  {{.*}}Ej
! CHECK:  %[[VAL_7:.*]] = fir.alloca i64 {bindc_name = "step", uniq_name = "_QFsubstringEstep"}
! CHECK:  %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7:[a-z0-9]*]]  {{.*}}Estep
! CHECK:  %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_1:[a-z0-9]*]]  {{.*}}Evector
! CHECK:  %[[VAL_10:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_11:.*]]:3 = fir.box_dims %[[VAL_9]]#0, %[[VAL_10]] : (!fir.box<!fir.array<?xi64>>, index) -> (index, index, index)
! CHECK:  %[[VAL_12:.*]] = fir.shape %[[VAL_11]]#1 : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_13:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_14:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_15:.*]] = fir.convert %[[VAL_13]] : (i64) -> index
! CHECK:  %[[VAL_16:.*]] = fir.convert %[[VAL_14]] : (i64) -> index
! CHECK:  %[[VAL_17:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_18:.*]] = arith.subi %[[VAL_16]], %[[VAL_15]] : index
! CHECK:  %[[VAL_19:.*]] = arith.addi %[[VAL_18]], %[[VAL_17]] : index
! CHECK:  %[[VAL_20:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_21:.*]] = arith.cmpi sgt, %[[VAL_19]], %[[VAL_20]] : index
! CHECK:  %[[VAL_22:.*]] = arith.select %[[VAL_21]], %[[VAL_19]], %[[VAL_20]] : index
! CHECK:  %[[VAL_23:.*]] = hlfir.elemental %[[VAL_12]] typeparams %[[VAL_22]] unordered : (!fir.shape<1>, index) -> !hlfir.expr<?x!fir.char<1,?>> {
! CHECK:  ^bb0(%[[VAL_24:.*]]: index):
! CHECK:    %[[VAL_25:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_24]])  : (!fir.box<!fir.array<?xi64>>, index) -> !fir.ref<i64>
! CHECK:    %[[VAL_26:.*]] = fir.load %[[VAL_25]] : !fir.ref<i64>
! CHECK:    %[[VAL_27:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_26]]) substr %[[VAL_15]], %[[VAL_16]]  typeparams %[[VAL_22]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, i64, index, index, index) -> !fir.boxchar<1>
! CHECK:    hlfir.yield_element %[[VAL_27]] : !fir.boxchar<1>
! CHECK:  }

subroutine test_passing_subscripted_poly(x, vector)
  interface
    subroutine do_something(x)
      class(*) :: x(:)
    end subroutine
  end interface
  class(*) :: x(:, :)
  integer(8) :: vector(:)
  call do_something(x(314, vector))
end subroutine
! CHECK-LABEL:   func.func @_QPtest_passing_subscripted_poly(
! CHECK-SAME:                                                %[[VAL_0:.*]]: !fir.class<!fir.array<?x?xnone>>
! CHECK-SAME:                                                %[[VAL_1:.*]]: !fir.box<!fir.array<?xi64>>
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFtest_passing_subscripted_polyEvector"} : (!fir.box<!fir.array<?xi64>>) -> (!fir.box<!fir.array<?xi64>>, !fir.box<!fir.array<?xi64>>)
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest_passing_subscripted_polyEx"} : (!fir.class<!fir.array<?x?xnone>>) -> (!fir.class<!fir.array<?x?xnone>>, !fir.class<!fir.array<?x?xnone>>)
! CHECK:           %[[VAL_4:.*]] = arith.constant 314 : index
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_2]]#0, %[[VAL_5]] : (!fir.box<!fir.array<?xi64>>, index) -> (index, index, index)
! CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_6]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_8:.*]] = hlfir.elemental %[[VAL_7]] mold %[[VAL_3]]#0 unordered : (!fir.shape<1>, !fir.class<!fir.array<?x?xnone>>) -> !hlfir.expr<?xnone?> {
! CHECK:           ^bb0(%[[VAL_9:.*]]: index):
! CHECK:             %[[VAL_10:.*]] = hlfir.designate %[[VAL_2]]#0 (%[[VAL_9]])  : (!fir.box<!fir.array<?xi64>>, index) -> !fir.ref<i64>
! CHECK:             %[[VAL_11:.*]] = fir.load %[[VAL_10]] : !fir.ref<i64>
! CHECK:             %[[VAL_12:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_4]], %[[VAL_11]])  : (!fir.class<!fir.array<?x?xnone>>, index, i64) -> !fir.class<none>
! CHECK:             hlfir.yield_element %[[VAL_12]] : !fir.class<none>
! CHECK:           }
! CHECK:           %[[VAL_13:.*]]:3 = hlfir.associate %[[VAL_8]](%[[VAL_7]]) {adapt.valuebyref} : (!hlfir.expr<?xnone?>, !fir.shape<1>) -> (!fir.class<!fir.heap<!fir.array<?xnone>>>, !fir.class<!fir.heap<!fir.array<?xnone>>>, i1)
! CHECK:           %[[VAL_14:.*]] = fir.rebox %[[VAL_13]]#0 : (!fir.class<!fir.heap<!fir.array<?xnone>>>) -> !fir.class<!fir.array<?xnone>>
! CHECK:           fir.call @_QPdo_something(%[[VAL_14]]) fastmath<contract> : (!fir.class<!fir.array<?xnone>>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_13]]#0, %[[VAL_13]]#2 : !fir.class<!fir.heap<!fir.array<?xnone>>>, i1
! CHECK:           hlfir.destroy %[[VAL_8]] : !hlfir.expr<?xnone?>
! CHECK:           return
! CHECK:         }
