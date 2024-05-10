! Test lowering of array constructors as inlined temporary.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine test_simple(i)
  call takes_int([42, i])
end subroutine
! CHECK-LABEL: func.func @_QPtest_simple(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare {{.*}}Ei
! CHECK:  %[[VAL_2:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_3:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_3B:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_4:.*]] = fir.allocmem !fir.array<2xi32> {bindc_name = ".tmp.arrayctor", uniq_name = ""}
! CHECK:  %[[VAL_5:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_4]](%[[VAL_5]]) {uniq_name = ".tmp.arrayctor"} : (!fir.heap<!fir.array<2xi32>>, !fir.shape<1>) -> (!fir.heap<!fir.array<2xi32>>, !fir.heap<!fir.array<2xi32>>)
! CHECK:  %[[VAL_7:.*]] = arith.constant 42 : i32
! CHECK:  %[[VAL_8:.*]] = arith.addi %[[VAL_3]], %[[VAL_3B]] : index
! CHECK:  %[[VAL_9:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_3]])  : (!fir.heap<!fir.array<2xi32>>, index) -> !fir.ref<i32>
! CHECK:  hlfir.assign %[[VAL_7]] to %[[VAL_9]] : i32, !fir.ref<i32>
! CHECK:  %[[VAL_10:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_11:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_8]])  : (!fir.heap<!fir.array<2xi32>>, index) -> !fir.ref<i32>
! CHECK:  hlfir.assign %[[VAL_10]] to %[[VAL_11]] : i32, !fir.ref<i32>
! CHECK:  %[[VAL_12:.*]] = arith.constant true
! CHECK:  %[[VAL_13:.*]] = hlfir.as_expr %[[VAL_6]]#0 move %[[VAL_12]] : (!fir.heap<!fir.array<2xi32>>, i1) -> !hlfir.expr<2xi32>
! CHECK:  fir.call
! CHECK:  hlfir.destroy %[[VAL_13]] : !hlfir.expr<2xi32>

subroutine test_simple_real(x)
  real(2) :: x
  call takes_real_2([x, 0._2])
end subroutine
! CHECK-LABEL: func.func @_QPtest_simple_real(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare {{.*}}Ex
! CHECK:  %[[VAL_2:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_3:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_3B:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_4:.*]] = fir.allocmem !fir.array<2xf16> {bindc_name = ".tmp.arrayctor", uniq_name = ""}
! CHECK:  %[[VAL_5:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_4]](%[[VAL_5]]) {uniq_name = ".tmp.arrayctor"} : (!fir.heap<!fir.array<2xf16>>, !fir.shape<1>) -> (!fir.heap<!fir.array<2xf16>>, !fir.heap<!fir.array<2xf16>>)
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<f16>
! CHECK:  %[[VAL_8:.*]] = arith.addi %[[VAL_3]], %[[VAL_3B]] : index
! CHECK:  %[[VAL_9:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_3]])  : (!fir.heap<!fir.array<2xf16>>, index) -> !fir.ref<f16>
! CHECK:  hlfir.assign %[[VAL_7]] to %[[VAL_9]] : f16, !fir.ref<f16>
! CHECK:  %[[VAL_10:.*]] = arith.constant 0.000000e+00 : f16
! CHECK:  %[[VAL_11:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_8]])  : (!fir.heap<!fir.array<2xf16>>, index) -> !fir.ref<f16>
! CHECK:  hlfir.assign %[[VAL_10]] to %[[VAL_11]] : f16, !fir.ref<f16>
! CHECK:  %[[VAL_12:.*]] = arith.constant true
! CHECK:  %[[VAL_13:.*]] = hlfir.as_expr %[[VAL_6]]#0 move %[[VAL_12]] : (!fir.heap<!fir.array<2xf16>>, i1) -> !hlfir.expr<2xf16>
! CHECK:  fir.call
! CHECK:  hlfir.destroy %[[VAL_13]] : !hlfir.expr<2xf16>

subroutine test_simple_complex(z)
  complex :: z
  call takes_cmplx_8([complex(8):: 42, z])
end subroutine
! CHECK-LABEL: func.func @_QPtest_simple_complex(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare {{.*}}Ez
! CHECK:  %[[VAL_2:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_3:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_3B:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_4:.*]] = fir.allocmem !fir.array<2x!fir.complex<8>> {bindc_name = ".tmp.arrayctor", uniq_name = ""}
! CHECK:  %[[VAL_5:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_4]](%[[VAL_5]]) {uniq_name = ".tmp.arrayctor"} : (!fir.heap<!fir.array<2x!fir.complex<8>>>, !fir.shape<1>) -> (!fir.heap<!fir.array<2x!fir.complex<8>>>, !fir.heap<!fir.array<2x!fir.complex<8>>>)
! CHECK:  %[[VAL_7:.*]] = arith.constant 42 : i32
! CHECK:  %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> f64
! CHECK:  %[[VAL_9:.*]] = arith.constant 0.000000e+00 : f64
! CHECK:  %[[VAL_10:.*]] = fir.undefined !fir.complex<8>
! CHECK:  %[[VAL_11:.*]] = fir.insert_value %[[VAL_10]], %[[VAL_8]], [0 : index] : (!fir.complex<8>, f64) -> !fir.complex<8>
! CHECK:  %[[VAL_12:.*]] = fir.insert_value %[[VAL_11]], %[[VAL_9]], [1 : index] : (!fir.complex<8>, f64) -> !fir.complex<8>
! CHECK:  %[[VAL_13:.*]] = arith.addi %[[VAL_3]], %[[VAL_3B]] : index
! CHECK:  %[[VAL_14:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_3]])  : (!fir.heap<!fir.array<2x!fir.complex<8>>>, index) -> !fir.ref<!fir.complex<8>>
! CHECK:  hlfir.assign %[[VAL_12]] to %[[VAL_14]] : !fir.complex<8>, !fir.ref<!fir.complex<8>>
! CHECK:  %[[VAL_15:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.complex<4>>
! CHECK:  %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (!fir.complex<4>) -> !fir.complex<8>
! CHECK:  %[[VAL_17:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_13]])  : (!fir.heap<!fir.array<2x!fir.complex<8>>>, index) -> !fir.ref<!fir.complex<8>>
! CHECK:  hlfir.assign %[[VAL_16]] to %[[VAL_17]] : !fir.complex<8>, !fir.ref<!fir.complex<8>>
! CHECK:  %[[VAL_18:.*]] = arith.constant true
! CHECK:  %[[VAL_19:.*]] = hlfir.as_expr %[[VAL_6]]#0 move %[[VAL_18]] : (!fir.heap<!fir.array<2x!fir.complex<8>>>, i1) -> !hlfir.expr<2x!fir.complex<8>>
! CHECK:  fir.call
! CHECK:  hlfir.destroy %[[VAL_19]] : !hlfir.expr<2x!fir.complex<8>>

subroutine test_simple_logical(a, b)
  logical :: a, b
  call takes_logical([a, a.and.b])
end subroutine
! CHECK-LABEL: func.func @_QPtest_simple_logical(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare {{.*}}Ea
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare {{.*}}Eb
! CHECK:  %[[VAL_4:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_5:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_5B:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_6:.*]] = fir.allocmem !fir.array<2x!fir.logical<4>> {bindc_name = ".tmp.arrayctor", uniq_name = ""}
! CHECK:  %[[VAL_7:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_6]](%[[VAL_7]]) {uniq_name = ".tmp.arrayctor"} : (!fir.heap<!fir.array<2x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.heap<!fir.array<2x!fir.logical<4>>>, !fir.heap<!fir.array<2x!fir.logical<4>>>)
! CHECK:  %[[VAL_9:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:  %[[VAL_10:.*]] = arith.addi %[[VAL_5]], %[[VAL_5B]] : index
! CHECK:  %[[VAL_11:.*]] = hlfir.designate %[[VAL_8]]#0 (%[[VAL_5]])  : (!fir.heap<!fir.array<2x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
! CHECK:  hlfir.assign %[[VAL_9]] to %[[VAL_11]] : !fir.logical<4>, !fir.ref<!fir.logical<4>>
! CHECK:  %[[VAL_12:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:  %[[VAL_13:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:  %[[VAL_14:.*]] = fir.convert %[[VAL_12]] : (!fir.logical<4>) -> i1
! CHECK:  %[[VAL_15:.*]] = fir.convert %[[VAL_13]] : (!fir.logical<4>) -> i1
! CHECK:  %[[VAL_16:.*]] = arith.andi %[[VAL_14]], %[[VAL_15]] : i1
! CHECK:  %[[VAL_17:.*]] = hlfir.designate %[[VAL_8]]#0 (%[[VAL_10]])  : (!fir.heap<!fir.array<2x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
! CHECK:  hlfir.assign %[[VAL_16]] to %[[VAL_17]] : i1, !fir.ref<!fir.logical<4>>
! CHECK:  %[[VAL_18:.*]] = arith.constant true
! CHECK:  %[[VAL_19:.*]] = hlfir.as_expr %[[VAL_8]]#0 move %[[VAL_18]] : (!fir.heap<!fir.array<2x!fir.logical<4>>>, i1) -> !hlfir.expr<2x!fir.logical<4>>
! CHECK:  fir.call
! CHECK:  hlfir.destroy %[[VAL_19]] : !hlfir.expr<2x!fir.logical<4>>

subroutine test_implied_do(n)
  integer(8) :: n
  ! This implied do cannot easily be promoted to hlfir.elemental because
  ! the implied do contains more than one scalar ac-value.
  call takes_int([(42, j, j=1,n)])
end subroutine
! CHECK-LABEL:   func.func @_QPtest_implied_do(
! CHECK-SAME:                                  %[[VAL_0:.*]]: !fir.ref<i64> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca index
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest_implied_doEn"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_4:.*]] = arith.constant 2 : i64
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<i64>
! CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_7:.*]] = arith.subi %[[VAL_5]], %[[VAL_6]] : i64
! CHECK:           %[[VAL_8:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_9:.*]] = arith.addi %[[VAL_7]], %[[VAL_8]] : i64
! CHECK:           %[[VAL_10:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_11:.*]] = arith.divsi %[[VAL_9]], %[[VAL_10]] : i64
! CHECK:           %[[VAL_12:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_13:.*]] = arith.cmpi sgt, %[[VAL_11]], %[[VAL_12]] : i64
! CHECK:           %[[VAL_14:.*]] = arith.select %[[VAL_13]], %[[VAL_11]], %[[VAL_12]] : i64
! CHECK:           %[[VAL_15:.*]] = arith.muli %[[VAL_4]], %[[VAL_14]] : i64
! CHECK:           %[[VAL_16:.*]] = arith.addi %[[VAL_3]], %[[VAL_15]] : i64
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i64) -> index
! CHECK:           %[[VAL_18:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_18B:.*]] = arith.constant 1 : index
! CHECK:           fir.store %[[VAL_18]] to %[[VAL_1]] : !fir.ref<index>
! CHECK:           %[[VAL_19:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_17]] {bindc_name = ".tmp.arrayctor", uniq_name = ""}
! CHECK:           %[[VAL_20:.*]] = fir.shape %[[VAL_17]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_21:.*]]:2 = hlfir.declare %[[VAL_19]](%[[VAL_20]]) {uniq_name = ".tmp.arrayctor"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! CHECK:           %[[VAL_22:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i64) -> index
! CHECK:           %[[VAL_24:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<i64>
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (i64) -> index
! CHECK:           %[[VAL_26:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i64) -> index
! CHECK:           fir.do_loop %[[VAL_28:.*]] = %[[VAL_23]] to %[[VAL_25]] step %[[VAL_27]] {
! CHECK:             %[[VAL_29:.*]] = arith.constant 42 : i32
! CHECK:             %[[VAL_30:.*]] = fir.load %[[VAL_1]] : !fir.ref<index>
! CHECK:             %[[VAL_31:.*]] = arith.addi %[[VAL_30]], %[[VAL_18B]] : index
! CHECK:             fir.store %[[VAL_31]] to %[[VAL_1]] : !fir.ref<index>
! CHECK:             %[[VAL_32:.*]] = hlfir.designate %[[VAL_21]]#0 (%[[VAL_30]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:             hlfir.assign %[[VAL_29]] to %[[VAL_32]] : i32, !fir.ref<i32>
! CHECK:             %[[VAL_33:.*]] = fir.convert %[[VAL_28]] : (index) -> i64
! CHECK:             %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (i64) -> i32
! CHECK:             %[[VAL_35:.*]] = fir.load %[[VAL_1]] : !fir.ref<index>
! CHECK:             %[[VAL_36:.*]] = arith.addi %[[VAL_35]], %[[VAL_18B]] : index
! CHECK:             fir.store %[[VAL_36]] to %[[VAL_1]] : !fir.ref<index>
! CHECK:             %[[VAL_37:.*]] = hlfir.designate %[[VAL_21]]#0 (%[[VAL_35]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:             hlfir.assign %[[VAL_34]] to %[[VAL_37]] : i32, !fir.ref<i32>
! CHECK:           }
! CHECK:           %[[VAL_38:.*]] = arith.constant true
! CHECK:           %[[VAL_39:.*]] = hlfir.as_expr %[[VAL_21]]#0 move %[[VAL_38]] : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<?xi32>
! CHECK:           %[[VAL_40:.*]]:3 = hlfir.associate %[[VAL_39]](%[[VAL_20]]) {adapt.valuebyref} : (!hlfir.expr<?xi32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>, i1)
! CHECK:           %[[VAL_41:.*]] = fir.convert %[[VAL_40]]#1 : (!fir.ref<!fir.array<?xi32>>) -> !fir.ref<!fir.array<2xi32>>
! CHECK:           fir.call @_QPtakes_int(%[[VAL_41]]) fastmath<contract> : (!fir.ref<!fir.array<2xi32>>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_40]]#1, %[[VAL_40]]#2 : !fir.ref<!fir.array<?xi32>>, i1
! CHECK:           hlfir.destroy %[[VAL_39]] : !hlfir.expr<?xi32>
! CHECK:           return
! CHECK:         }

subroutine test_strided_implied_do(lb, ub, stride)
  integer(8) :: lb, ub, stride
  call takes_int([(42, j, j=lb,ub,stride)])
end subroutine
! CHECK-LABEL:   func.func @_QPtest_strided_implied_do(
! CHECK-SAME:                                          %[[VAL_0:.*]]: !fir.ref<i64> {fir.bindc_name = "lb"},
! CHECK-SAME:                                          %[[VAL_1:.*]]: !fir.ref<i64> {fir.bindc_name = "ub"},
! CHECK-SAME:                                          %[[VAL_2:.*]]: !fir.ref<i64> {fir.bindc_name = "stride"}) {
! CHECK:           %[[VAL_3:.*]] = fir.alloca index
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest_strided_implied_doElb"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFtest_strided_implied_doEstride"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFtest_strided_implied_doEub"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[VAL_7:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_8:.*]] = arith.constant 2 : i64
! CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<i64>
! CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i64>
! CHECK:           %[[VAL_11:.*]] = arith.subi %[[VAL_9]], %[[VAL_10]] : i64
! CHECK:           %[[VAL_12:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i64>
! CHECK:           %[[VAL_13:.*]] = arith.addi %[[VAL_11]], %[[VAL_12]] : i64
! CHECK:           %[[VAL_14:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i64>
! CHECK:           %[[VAL_15:.*]] = arith.divsi %[[VAL_13]], %[[VAL_14]] : i64
! CHECK:           %[[VAL_16:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_17:.*]] = arith.cmpi sgt, %[[VAL_15]], %[[VAL_16]] : i64
! CHECK:           %[[VAL_18:.*]] = arith.select %[[VAL_17]], %[[VAL_15]], %[[VAL_16]] : i64
! CHECK:           %[[VAL_19:.*]] = arith.muli %[[VAL_8]], %[[VAL_18]] : i64
! CHECK:           %[[VAL_20:.*]] = arith.addi %[[VAL_7]], %[[VAL_19]] : i64
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i64) -> index
! CHECK:           %[[VAL_22:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_22B:.*]] = arith.constant 1 : index
! CHECK:           fir.store %[[VAL_22]] to %[[VAL_3]] : !fir.ref<index>
! CHECK:           %[[VAL_23:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_21]] {bindc_name = ".tmp.arrayctor", uniq_name = ""}
! CHECK:           %[[VAL_24:.*]] = fir.shape %[[VAL_21]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_25:.*]]:2 = hlfir.declare %[[VAL_23]](%[[VAL_24]]) {uniq_name = ".tmp.arrayctor"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! CHECK:           %[[VAL_26:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i64>
! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i64) -> index
! CHECK:           %[[VAL_28:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<i64>
! CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (i64) -> index
! CHECK:           %[[VAL_30:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i64>
! CHECK:           %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (i64) -> index
! CHECK:           fir.do_loop %[[VAL_32:.*]] = %[[VAL_27]] to %[[VAL_29]] step %[[VAL_31]] {
! CHECK:             %[[VAL_33:.*]] = arith.constant 42 : i32
! CHECK:             %[[VAL_34:.*]] = fir.load %[[VAL_3]] : !fir.ref<index>
! CHECK:             %[[VAL_35:.*]] = arith.addi %[[VAL_34]], %[[VAL_22B]] : index
! CHECK:             fir.store %[[VAL_35]] to %[[VAL_3]] : !fir.ref<index>
! CHECK:             %[[VAL_36:.*]] = hlfir.designate %[[VAL_25]]#0 (%[[VAL_34]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:             hlfir.assign %[[VAL_33]] to %[[VAL_36]] : i32, !fir.ref<i32>
! CHECK:             %[[VAL_37:.*]] = fir.convert %[[VAL_32]] : (index) -> i64
! CHECK:             %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (i64) -> i32
! CHECK:             %[[VAL_39:.*]] = fir.load %[[VAL_3]] : !fir.ref<index>
! CHECK:             %[[VAL_40:.*]] = arith.addi %[[VAL_39]], %[[VAL_22B]] : index
! CHECK:             fir.store %[[VAL_40]] to %[[VAL_3]] : !fir.ref<index>
! CHECK:             %[[VAL_41:.*]] = hlfir.designate %[[VAL_25]]#0 (%[[VAL_39]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:             hlfir.assign %[[VAL_38]] to %[[VAL_41]] : i32, !fir.ref<i32>
! CHECK:           }
! CHECK:           %[[VAL_42:.*]] = arith.constant true
! CHECK:           %[[VAL_43:.*]] = hlfir.as_expr %[[VAL_25]]#0 move %[[VAL_42]] : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<?xi32>
! CHECK:           %[[VAL_44:.*]]:3 = hlfir.associate %[[VAL_43]](%[[VAL_24]]) {adapt.valuebyref} : (!hlfir.expr<?xi32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>, i1)
! CHECK:           %[[VAL_45:.*]] = fir.convert %[[VAL_44]]#1 : (!fir.ref<!fir.array<?xi32>>) -> !fir.ref<!fir.array<2xi32>>
! CHECK:           fir.call @_QPtakes_int(%[[VAL_45]]) fastmath<contract> : (!fir.ref<!fir.array<2xi32>>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_44]]#1, %[[VAL_44]]#2 : !fir.ref<!fir.array<?xi32>>, i1
! CHECK:           hlfir.destroy %[[VAL_43]] : !hlfir.expr<?xi32>
! CHECK:           return
! CHECK:         }

subroutine test_nested_implied_do(n, m)
  integer(8) :: n, m
  call takes_int([((i+j, i=1,m), j=1,n)])
end subroutine
! CHECK-LABEL:   func.func @_QPtest_nested_implied_do(
! CHECK-SAME:                                         %[[VAL_0:.*]]: !fir.ref<i64> {fir.bindc_name = "n"},
! CHECK-SAME:                                         %[[VAL_1:.*]]: !fir.ref<i64> {fir.bindc_name = "m"}) {
! CHECK:           %[[VAL_2:.*]] = fir.alloca index
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFtest_nested_implied_doEm"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest_nested_implied_doEn"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i64>
! CHECK:           %[[VAL_8:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_9:.*]] = arith.subi %[[VAL_7]], %[[VAL_8]] : i64
! CHECK:           %[[VAL_10:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_11:.*]] = arith.addi %[[VAL_9]], %[[VAL_10]] : i64
! CHECK:           %[[VAL_12:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_13:.*]] = arith.divsi %[[VAL_11]], %[[VAL_12]] : i64
! CHECK:           %[[VAL_14:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_15:.*]] = arith.cmpi sgt, %[[VAL_13]], %[[VAL_14]] : i64
! CHECK:           %[[VAL_16:.*]] = arith.select %[[VAL_15]], %[[VAL_13]], %[[VAL_14]] : i64
! CHECK:           %[[VAL_17:.*]] = arith.addi %[[VAL_6]], %[[VAL_16]] : i64
! CHECK:           %[[VAL_18:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i64>
! CHECK:           %[[VAL_19:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_20:.*]] = arith.subi %[[VAL_18]], %[[VAL_19]] : i64
! CHECK:           %[[VAL_21:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_22:.*]] = arith.addi %[[VAL_20]], %[[VAL_21]] : i64
! CHECK:           %[[VAL_23:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_24:.*]] = arith.divsi %[[VAL_22]], %[[VAL_23]] : i64
! CHECK:           %[[VAL_25:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_26:.*]] = arith.cmpi sgt, %[[VAL_24]], %[[VAL_25]] : i64
! CHECK:           %[[VAL_27:.*]] = arith.select %[[VAL_26]], %[[VAL_24]], %[[VAL_25]] : i64
! CHECK:           %[[VAL_28:.*]] = arith.muli %[[VAL_17]], %[[VAL_27]] : i64
! CHECK:           %[[VAL_29:.*]] = arith.addi %[[VAL_5]], %[[VAL_28]] : i64
! CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i64) -> index
! CHECK:           %[[VAL_31:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_31B:.*]] = arith.constant 1 : index
! CHECK:           fir.store %[[VAL_31]] to %[[VAL_2]] : !fir.ref<index>
! CHECK:           %[[VAL_32:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_30]] {bindc_name = ".tmp.arrayctor", uniq_name = ""}
! CHECK:           %[[VAL_33:.*]] = fir.shape %[[VAL_30]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_34:.*]]:2 = hlfir.declare %[[VAL_32]](%[[VAL_33]]) {uniq_name = ".tmp.arrayctor"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! CHECK:           %[[VAL_35:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (i64) -> index
! CHECK:           %[[VAL_37:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i64>
! CHECK:           %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (i64) -> index
! CHECK:           %[[VAL_39:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_40:.*]] = fir.convert %[[VAL_39]] : (i64) -> index
! CHECK:           fir.do_loop %[[VAL_41:.*]] = %[[VAL_36]] to %[[VAL_38]] step %[[VAL_40]] {
! CHECK:             %[[VAL_42:.*]] = arith.constant 1 : i64
! CHECK:             %[[VAL_43:.*]] = fir.convert %[[VAL_42]] : (i64) -> index
! CHECK:             %[[VAL_44:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i64>
! CHECK:             %[[VAL_45:.*]] = fir.convert %[[VAL_44]] : (i64) -> index
! CHECK:             %[[VAL_46:.*]] = arith.constant 1 : i64
! CHECK:             %[[VAL_47:.*]] = fir.convert %[[VAL_46]] : (i64) -> index
! CHECK:             fir.do_loop %[[VAL_48:.*]] = %[[VAL_43]] to %[[VAL_45]] step %[[VAL_47]] {
! CHECK:               %[[VAL_49:.*]] = fir.convert %[[VAL_48]] : (index) -> i64
! CHECK:               %[[VAL_50:.*]] = fir.convert %[[VAL_49]] : (i64) -> i32
! CHECK:               %[[VAL_51:.*]] = fir.convert %[[VAL_41]] : (index) -> i64
! CHECK:               %[[VAL_52:.*]] = fir.convert %[[VAL_51]] : (i64) -> i32
! CHECK:               %[[VAL_53:.*]] = arith.addi %[[VAL_50]], %[[VAL_52]] : i32
! CHECK:               %[[VAL_54:.*]] = fir.load %[[VAL_2]] : !fir.ref<index>
! CHECK:               %[[VAL_55:.*]] = arith.addi %[[VAL_54]], %[[VAL_31B]] : index
! CHECK:               fir.store %[[VAL_55]] to %[[VAL_2]] : !fir.ref<index>
! CHECK:               %[[VAL_56:.*]] = hlfir.designate %[[VAL_34]]#0 (%[[VAL_54]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:               hlfir.assign %[[VAL_53]] to %[[VAL_56]] : i32, !fir.ref<i32>
! CHECK:             }
! CHECK:           }
! CHECK:           %[[VAL_57:.*]] = arith.constant true
! CHECK:           %[[VAL_58:.*]] = hlfir.as_expr %[[VAL_34]]#0 move %[[VAL_57]] : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<?xi32>
! CHECK:           %[[VAL_59:.*]]:3 = hlfir.associate %[[VAL_58]](%[[VAL_33]]) {adapt.valuebyref} : (!hlfir.expr<?xi32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>, i1)
! CHECK:           %[[VAL_60:.*]] = fir.convert %[[VAL_59]]#1 : (!fir.ref<!fir.array<?xi32>>) -> !fir.ref<!fir.array<2xi32>>
! CHECK:           fir.call @_QPtakes_int(%[[VAL_60]]) fastmath<contract> : (!fir.ref<!fir.array<2xi32>>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_59]]#1, %[[VAL_59]]#2 : !fir.ref<!fir.array<?xi32>>, i1
! CHECK:           hlfir.destroy %[[VAL_58]] : !hlfir.expr<?xi32>
! CHECK:           return
! CHECK:         }
