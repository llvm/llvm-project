! Test lowering of vector subscripted designators in assignment
! left-hand sides.
! RUN: bbc -emit-hlfir -o - -I nw %s 2>&1 | FileCheck %s

subroutine test_simple(x, vector)
  integer(8) :: vector(10)
  real :: x(:)
  x(vector) = 42.
end subroutine
! CHECK-LABEL:   func.func @_QPtest_simple(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare {{.*}}Evector
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare {{.*}}Ex
! CHECK:  hlfir.region_assign {
! CHECK:    %[[VAL_6:.*]] = arith.constant 4.200000e+01 : f32
! CHECK:    hlfir.yield %[[VAL_6]] : f32
! CHECK:  } to {
! CHECK:    %[[VAL_7:.*]] = arith.constant 10 : index
! CHECK:    %[[VAL_8:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:    hlfir.elemental_addr %[[VAL_8]] unordered : !fir.shape<1> {
! CHECK:    ^bb0(%[[VAL_9:.*]]: index):
! CHECK:      %[[VAL_10:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_9]])  : (!fir.ref<!fir.array<10xi64>>, index) -> !fir.ref<i64>
! CHECK:      %[[VAL_11:.*]] = fir.load %[[VAL_10]] : !fir.ref<i64>
! CHECK:      %[[VAL_12:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_11]])  : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
! CHECK:      hlfir.yield %[[VAL_12]] : !fir.ref<f32>
! CHECK:    }
! CHECK:  }

subroutine test_cleanup(x, vector, matrix)
  integer(8) :: vector(10), matrix(10, 5)
  real :: x(:)
  x(matmul(vector, matrix)) = 42.
end subroutine
! CHECK-LABEL:   func.func @_QPtest_cleanup(
! CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare {{.*}}Ematrix
! CHECK:  %[[VAL_9:.*]]:2 = hlfir.declare {{.*}}Evector
! CHECK:  %[[VAL_10:.*]]:2 = hlfir.declare {{.*}}Ex
! CHECK:  hlfir.region_assign {
! CHECK:    %[[VAL_11:.*]] = arith.constant 4.200000e+01 : f32
! CHECK:    hlfir.yield %[[VAL_11]] : f32
! CHECK:  } to {
! CHECK:    %[[VAL_12:.*]] = hlfir.matmul %[[VAL_9]]#0 %[[VAL_6]]#0 {fastmath = #arith.fastmath<contract>} : (!fir.ref<!fir.array<10xi64>>, !fir.ref<!fir.array<10x5xi64>>) -> !hlfir.expr<5xi64>
! CHECK:    %[[VAL_13:.*]] = arith.constant 5 : index
! CHECK:    %[[VAL_14:.*]] = fir.shape %[[VAL_13]] : (index) -> !fir.shape<1>
! CHECK:    hlfir.elemental_addr %[[VAL_14]] unordered : !fir.shape<1> {
! CHECK:    ^bb0(%[[VAL_15:.*]]: index):
! CHECK:      %[[VAL_16:.*]] = hlfir.apply %[[VAL_12]], %[[VAL_15]] : (!hlfir.expr<5xi64>, index) -> i64
! CHECK:      %[[VAL_17:.*]] = hlfir.designate %[[VAL_10]]#0 (%[[VAL_16]])  : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
! CHECK:      hlfir.yield %[[VAL_17]] : !fir.ref<f32>
! CHECK:    } cleanup {
! CHECK:      hlfir.destroy %[[VAL_12]] : !hlfir.expr<5xi64>
! CHECK:    }
! CHECK:  }

subroutine test_nested_vectors(x, vector1, vector2, vector3)
  integer(8) :: vector1(10), vector2(8), vector3(6)
  real :: x(:)
  x(vector1(vector2(vector3))) = 42.
end subroutine
! CHECK-LABEL:   func.func @_QPtest_nested_vectors(
! CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare {{.*}}Evector1
! CHECK:  %[[VAL_9:.*]]:2 = hlfir.declare {{.*}}Evector2
! CHECK:  %[[VAL_12:.*]]:2 = hlfir.declare {{.*}}Evector3
! CHECK:  %[[VAL_13:.*]]:2 = hlfir.declare {{.*}}Ex
! CHECK:  hlfir.region_assign {
! CHECK:    %[[VAL_14:.*]] = arith.constant 4.200000e+01 : f32
! CHECK:    hlfir.yield %[[VAL_14]] : f32
! CHECK:  } to {
! CHECK:    %[[VAL_15:.*]] = arith.constant 6 : index
! CHECK:    %[[VAL_16:.*]] = fir.shape %[[VAL_15]] : (index) -> !fir.shape<1>
! CHECK:    %[[VAL_17:.*]] = hlfir.elemental %[[VAL_16]] unordered : (!fir.shape<1>) -> !hlfir.expr<6xi64> {
! CHECK:    ^bb0(%[[VAL_18:.*]]: index):
! CHECK:      %[[VAL_19:.*]] = hlfir.designate %[[VAL_12]]#0 (%[[VAL_18]])  : (!fir.ref<!fir.array<6xi64>>, index) -> !fir.ref<i64>
! CHECK:      %[[VAL_20:.*]] = fir.load %[[VAL_19]] : !fir.ref<i64>
! CHECK:      %[[VAL_21:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_20]])  : (!fir.ref<!fir.array<8xi64>>, i64) -> !fir.ref<i64>
! CHECK:      %[[VAL_22:.*]] = fir.load %[[VAL_21]] : !fir.ref<i64>
! CHECK:      hlfir.yield_element %[[VAL_22]] : i64
! CHECK:    }
! CHECK:    %[[VAL_23:.*]] = arith.constant 6 : index
! CHECK:    %[[VAL_24:.*]] = fir.shape %[[VAL_23]] : (index) -> !fir.shape<1>
! CHECK:    %[[VAL_25:.*]] = hlfir.elemental %[[VAL_24]] unordered : (!fir.shape<1>) -> !hlfir.expr<6xi64> {
! CHECK:    ^bb0(%[[VAL_26:.*]]: index):
! CHECK:      %[[VAL_27:.*]] = hlfir.apply %[[VAL_28:.*]], %[[VAL_26]] : (!hlfir.expr<6xi64>, index) -> i64
! CHECK:      %[[VAL_29:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_27]])  : (!fir.ref<!fir.array<10xi64>>, i64) -> !fir.ref<i64>
! CHECK:      %[[VAL_30:.*]] = fir.load %[[VAL_29]] : !fir.ref<i64>
! CHECK:      hlfir.yield_element %[[VAL_30]] : i64
! CHECK:    }
! CHECK:    %[[VAL_31:.*]] = arith.constant 6 : index
! CHECK:    %[[VAL_32:.*]] = fir.shape %[[VAL_31]] : (index) -> !fir.shape<1>
! CHECK:    hlfir.elemental_addr %[[VAL_32]] unordered : !fir.shape<1> {
! CHECK:    ^bb0(%[[VAL_33:.*]]: index):
! CHECK:      %[[VAL_34:.*]] = hlfir.apply %[[VAL_35:.*]], %[[VAL_33]] : (!hlfir.expr<6xi64>, index) -> i64
! CHECK:      %[[VAL_36:.*]] = hlfir.designate %[[VAL_13]]#0 (%[[VAL_34]])  : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
! CHECK:      hlfir.yield %[[VAL_36]] : !fir.ref<f32>
! CHECK:    } cleanup {
! CHECK:      hlfir.destroy %[[VAL_37:.*]] : !hlfir.expr<6xi64>
! CHECK:      hlfir.destroy %[[VAL_38:.*]] : !hlfir.expr<6xi64>
! CHECK:    }
! CHECK:  }

subroutine test_substring(x, vector)
  integer(8) :: vector(10), ifoo, ibar
  external :: ifoo, ibar
  character(*) :: x(:)
  x(vector)(ifoo(): ibar()) = "hello"
end subroutine
! CHECK-LABEL:   func.func @_QPtest_substring(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare {{.*}}Evector
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare {{.*}}Ex
! CHECK:  hlfir.region_assign {
! CHECK:    %[[VAL_6:.*]] = fir.address_of(@{{.*}}) : !fir.ref<!fir.char<1,5>>
! CHECK:    %[[VAL_7:.*]] = arith.constant 5 : index
! CHECK:    %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_6]] typeparams %[[VAL_7]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX68656C6C6F"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
! CHECK:    hlfir.yield %[[VAL_8]]#0 : !fir.ref<!fir.char<1,5>>
! CHECK:  } to {
! CHECK:    %[[VAL_9:.*]] = arith.constant 10 : index
! CHECK:    %[[VAL_10:.*]] = fir.shape %[[VAL_9]] : (index) -> !fir.shape<1>
! CHECK:    %[[VAL_11:.*]] = fir.call @_QPifoo() {{.*}}: () -> i64
! CHECK:    %[[VAL_12:.*]] = fir.call @_QPibar() {{.*}}: () -> i64
! CHECK:    %[[VAL_13:.*]] = fir.convert %[[VAL_11]] : (i64) -> index
! CHECK:    %[[VAL_14:.*]] = fir.convert %[[VAL_12]] : (i64) -> index
! CHECK:    %[[VAL_15:.*]] = arith.constant 1 : index
! CHECK:    %[[VAL_16:.*]] = arith.subi %[[VAL_14]], %[[VAL_13]] : index
! CHECK:    %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_15]] : index
! CHECK:    %[[VAL_18:.*]] = arith.constant 0 : index
! CHECK:    %[[VAL_19:.*]] = arith.cmpi sgt, %[[VAL_17]], %[[VAL_18]] : index
! CHECK:    %[[VAL_20:.*]] = arith.select %[[VAL_19]], %[[VAL_17]], %[[VAL_18]] : index
! CHECK:    hlfir.elemental_addr %[[VAL_10]] typeparams %[[VAL_20]] unordered : !fir.shape<1>, index {
! CHECK:    ^bb0(%[[VAL_21:.*]]: index):
! CHECK:      %[[VAL_22:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_21]])  : (!fir.ref<!fir.array<10xi64>>, index) -> !fir.ref<i64>
! CHECK:      %[[VAL_23:.*]] = fir.load %[[VAL_22]] : !fir.ref<i64>
! CHECK:      %[[VAL_24:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_23]]) substr %[[VAL_13]], %[[VAL_14]]  typeparams %[[VAL_20]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, i64, index, index, index) -> !fir.boxchar<1>
! CHECK:      hlfir.yield %[[VAL_24]] : !fir.boxchar<1>
! CHECK:    }
! CHECK:  }

subroutine test_hard_array_ref(x, vector1, vector2)
  integer(8) :: vector1(10), vector2(20), ifoo, ibar, ibaz
  external :: ifoo, ibar, ibaz
  real :: x(:, :, :, :, :)
  x(vector1, :, ifoo():ibar(), ibaz(), vector2) = 42.
end subroutine
! CHECK-LABEL:   func.func @_QPtest_hard_array_ref(
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare {{.*}}Evector1
! CHECK:  %[[VAL_8:.*]]:2 = hlfir.declare {{.*}}Evector2
! CHECK:  %[[VAL_9:.*]]:2 = hlfir.declare {{.*}}Ex
! CHECK:  hlfir.region_assign {
! CHECK:    %[[VAL_10:.*]] = arith.constant 4.200000e+01 : f32
! CHECK:    hlfir.yield %[[VAL_10]] : f32
! CHECK:  } to {
! CHECK:    %[[VAL_11:.*]] = arith.constant 10 : index
! CHECK:    %[[VAL_12:.*]] = arith.constant 1 : index
! CHECK:    %[[VAL_13:.*]] = arith.constant 1 : index
! CHECK:    %[[VAL_14:.*]]:3 = fir.box_dims %[[VAL_9]]#1, %[[VAL_13]] : (!fir.box<!fir.array<?x?x?x?x?xf32>>, index) -> (index, index, index)
! CHECK:    %[[VAL_15:.*]] = arith.constant 1 : index
! CHECK:    %[[VAL_16:.*]] = arith.constant 0 : index
! CHECK:    %[[VAL_17:.*]] = arith.subi %[[VAL_14]]#1, %[[VAL_12]] : index
! CHECK:    %[[VAL_18:.*]] = arith.addi %[[VAL_17]], %[[VAL_15]] : index
! CHECK:    %[[VAL_19:.*]] = arith.divsi %[[VAL_18]], %[[VAL_15]] : index
! CHECK:    %[[VAL_20:.*]] = arith.cmpi sgt, %[[VAL_19]], %[[VAL_16]] : index
! CHECK:    %[[VAL_21:.*]] = arith.select %[[VAL_20]], %[[VAL_19]], %[[VAL_16]] : index
! CHECK:    %[[VAL_22:.*]] = fir.call @_QPifoo() {{.*}}: () -> i64
! CHECK:    %[[VAL_23:.*]] = fir.call @_QPibar() {{.*}}: () -> i64
! CHECK:    %[[VAL_24:.*]] = fir.convert %[[VAL_22]] : (i64) -> index
! CHECK:    %[[VAL_25:.*]] = fir.convert %[[VAL_23]] : (i64) -> index
! CHECK:    %[[VAL_26:.*]] = arith.constant 1 : index
! CHECK:    %[[VAL_27:.*]] = arith.constant 0 : index
! CHECK:    %[[VAL_28:.*]] = arith.subi %[[VAL_25]], %[[VAL_24]] : index
! CHECK:    %[[VAL_29:.*]] = arith.addi %[[VAL_28]], %[[VAL_26]] : index
! CHECK:    %[[VAL_30:.*]] = arith.divsi %[[VAL_29]], %[[VAL_26]] : index
! CHECK:    %[[VAL_31:.*]] = arith.cmpi sgt, %[[VAL_30]], %[[VAL_27]] : index
! CHECK:    %[[VAL_32:.*]] = arith.select %[[VAL_31]], %[[VAL_30]], %[[VAL_27]] : index
! CHECK:    %[[VAL_33:.*]] = fir.call @_QPibaz() {{.*}}: () -> i64
! CHECK:    %[[VAL_34:.*]] = arith.constant 20 : index
! CHECK:    %[[VAL_35:.*]] = fir.shape %[[VAL_11]], %[[VAL_21]], %[[VAL_32]], %[[VAL_34]] : (index, index, index, index) -> !fir.shape<4>
! CHECK:    hlfir.elemental_addr %[[VAL_35]] unordered : !fir.shape<4> {
! CHECK:    ^bb0(%[[VAL_36:.*]]: index, %[[VAL_37:.*]]: index, %[[VAL_38:.*]]: index, %[[VAL_39:.*]]: index):
! CHECK:      %[[VAL_40:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_36]])  : (!fir.ref<!fir.array<10xi64>>, index) -> !fir.ref<i64>
! CHECK:      %[[VAL_41:.*]] = fir.load %[[VAL_40]] : !fir.ref<i64>
! CHECK:      %[[VAL_42:.*]] = arith.constant 1 : index
! CHECK:      %[[VAL_43:.*]] = arith.subi %[[VAL_37]], %[[VAL_42]] : index
! CHECK:      %[[VAL_44:.*]] = arith.muli %[[VAL_43]], %[[VAL_15]] : index
! CHECK:      %[[VAL_45:.*]] = arith.addi %[[VAL_12]], %[[VAL_44]] : index
! CHECK:      %[[VAL_46:.*]] = arith.constant 1 : index
! CHECK:      %[[VAL_47:.*]] = arith.subi %[[VAL_38]], %[[VAL_46]] : index
! CHECK:      %[[VAL_48:.*]] = arith.muli %[[VAL_47]], %[[VAL_26]] : index
! CHECK:      %[[VAL_49:.*]] = arith.addi %[[VAL_24]], %[[VAL_48]] : index
! CHECK:      %[[VAL_50:.*]] = hlfir.designate %[[VAL_8]]#0 (%[[VAL_39]])  : (!fir.ref<!fir.array<20xi64>>, index) -> !fir.ref<i64>
! CHECK:      %[[VAL_51:.*]] = fir.load %[[VAL_50]] : !fir.ref<i64>
! CHECK:      %[[VAL_52:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_41]], %[[VAL_45]], %[[VAL_49]], %[[VAL_33]], %[[VAL_51]])  : (!fir.box<!fir.array<?x?x?x?x?xf32>>, i64, index, index, i64, i64) -> !fir.ref<f32>
! CHECK:      hlfir.yield %[[VAL_52]] : !fir.ref<f32>
! CHECK:    }
! CHECK:  }
! CHECK:  return
