// RUN: fir-opt %s --simplify-hlfir-intrinsics | FileCheck %s

func.func @test_1d_total_expr(%input: !hlfir.expr<?xi32>, %mask: !hlfir.expr<?x!fir.logical<4>>) -> !hlfir.expr<1xi32> {
  %0 = hlfir.minloc %input mask %mask {fastmath = #arith.fastmath<contract>} : (!hlfir.expr<?xi32>, !hlfir.expr<?x!fir.logical<4>>) -> !hlfir.expr<1xi32>
  return %0 : !hlfir.expr<1xi32>
}
// CHECK-LABEL:   func.func @test_1d_total_expr(
// CHECK-SAME:                                  %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !hlfir.expr<?xi32>,
// CHECK-SAME:                                  %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !hlfir.expr<?x!fir.logical<4>>) -> !hlfir.expr<1xi32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant false
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant true
// CHECK:           %[[VAL_5:.*]] = arith.constant 2147483647 : i32
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_7:.*]] = fir.alloca !fir.array<1xi32>
// CHECK:           %[[VAL_8:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?xi32>) -> !fir.shape<1>
// CHECK:           %[[VAL_9:.*]] = hlfir.get_extent %[[VAL_8]] {dim = 0 : index} : (!fir.shape<1>) -> index
// CHECK:           %[[VAL_10:.*]]:3 = fir.do_loop %[[VAL_11:.*]] = %[[VAL_3]] to %[[VAL_9]] step %[[VAL_3]] unordered iter_args(%[[VAL_12:.*]] = %[[VAL_6]], %[[VAL_13:.*]] = %[[VAL_5]], %[[VAL_14:.*]] = %[[VAL_4]]) -> (i32, i32, i1) {
// CHECK:             %[[VAL_15:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_11]] : (!hlfir.expr<?x!fir.logical<4>>, index) -> !fir.logical<4>
// CHECK:             %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (!fir.logical<4>) -> i1
// CHECK:             %[[VAL_17:.*]]:3 = fir.if %[[VAL_16]] -> (i32, i32, i1) {
// CHECK:               %[[VAL_18:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_11]] : (!hlfir.expr<?xi32>, index) -> i32
// CHECK:               %[[VAL_19:.*]] = arith.cmpi slt, %[[VAL_18]], %[[VAL_13]] : i32
// CHECK:               %[[VAL_20:.*]] = arith.ori %[[VAL_19]], %[[VAL_14]] : i1
// CHECK:               %[[VAL_21:.*]] = fir.convert %[[VAL_11]] : (index) -> i32
// CHECK:               %[[VAL_22:.*]] = arith.select %[[VAL_20]], %[[VAL_21]], %[[VAL_12]] : i32
// CHECK:               %[[VAL_23:.*]] = arith.select %[[VAL_20]], %[[VAL_18]], %[[VAL_13]] : i32
// CHECK:               fir.result %[[VAL_22]], %[[VAL_23]], %[[VAL_2]] : i32, i32, i1
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_12]], %[[VAL_13]], %[[VAL_14]] : i32, i32, i1
// CHECK:             }
// CHECK:             fir.result %[[VAL_24:.*]]#0, %[[VAL_24]]#1, %[[VAL_24]]#2 : i32, i32, i1
// CHECK:           }
// CHECK:           %[[VAL_28:.*]] = hlfir.designate %[[VAL_7]] (%[[VAL_3]])  : (!fir.ref<!fir.array<1xi32>>, index) -> !fir.ref<i32>
// CHECK:           hlfir.assign %[[VAL_10]]#0 to %[[VAL_28]] : i32, !fir.ref<i32>
// CHECK:           %[[VAL_29:.*]] = hlfir.as_expr %[[VAL_7]] move %[[VAL_2]] : (!fir.ref<!fir.array<1xi32>>, i1) -> !hlfir.expr<1xi32>
// CHECK:           return %[[VAL_29]] : !hlfir.expr<1xi32>
// CHECK:         }

func.func @test_1d_dim_expr(%input: !hlfir.expr<?xf32>, %mask: !hlfir.expr<?x!fir.logical<4>>) -> i32 {
  %dim = arith.constant 1 : i16
  %0 = hlfir.minloc %input dim %dim mask %mask {fastmath = #arith.fastmath<contract>} : (!hlfir.expr<?xf32>, i16, !hlfir.expr<?x!fir.logical<4>>) -> i32
  return %0 : i32
}
// CHECK-LABEL:   func.func @test_1d_dim_expr(
// CHECK-SAME:                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !hlfir.expr<?xf32>,
// CHECK-SAME:                                %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !hlfir.expr<?x!fir.logical<4>>) -> i32 {
// CHECK:           %[[VAL_2:.*]] = arith.constant false
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant true
// CHECK:           %[[VAL_5:.*]] = arith.constant 3.40282347E+38 : f32
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_7:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?xf32>) -> !fir.shape<1>
// CHECK:           %[[VAL_8:.*]] = hlfir.get_extent %[[VAL_7]] {dim = 0 : index} : (!fir.shape<1>) -> index
// CHECK:           %[[VAL_9:.*]]:3 = fir.do_loop %[[VAL_10:.*]] = %[[VAL_3]] to %[[VAL_8]] step %[[VAL_3]] iter_args(%[[VAL_11:.*]] = %[[VAL_6]], %[[VAL_12:.*]] = %[[VAL_5]], %[[VAL_13:.*]] = %[[VAL_4]]) -> (i32, f32, i1) {
// CHECK:             %[[VAL_14:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_10]] : (!hlfir.expr<?x!fir.logical<4>>, index) -> !fir.logical<4>
// CHECK:             %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (!fir.logical<4>) -> i1
// CHECK:             %[[VAL_16:.*]]:3 = fir.if %[[VAL_15]] -> (i32, f32, i1) {
// CHECK:               %[[VAL_17:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_10]] : (!hlfir.expr<?xf32>, index) -> f32
// CHECK:               %[[VAL_18:.*]] = arith.cmpf olt, %[[VAL_17]], %[[VAL_12]] fastmath<contract> : f32
// CHECK:               %[[VAL_19:.*]] = arith.cmpf une, %[[VAL_12]], %[[VAL_12]] fastmath<contract> : f32
// CHECK:               %[[VAL_20:.*]] = arith.cmpf oeq, %[[VAL_17]], %[[VAL_17]] fastmath<contract> : f32
// CHECK:               %[[VAL_21:.*]] = arith.andi %[[VAL_19]], %[[VAL_20]] : i1
// CHECK:               %[[VAL_22:.*]] = arith.ori %[[VAL_18]], %[[VAL_21]] : i1
// CHECK:               %[[VAL_23:.*]] = arith.ori %[[VAL_22]], %[[VAL_13]] : i1
// CHECK:               %[[VAL_24:.*]] = fir.convert %[[VAL_10]] : (index) -> i32
// CHECK:               %[[VAL_25:.*]] = arith.select %[[VAL_23]], %[[VAL_24]], %[[VAL_11]] : i32
// CHECK:               %[[VAL_26:.*]] = arith.select %[[VAL_23]], %[[VAL_17]], %[[VAL_12]] : f32
// CHECK:               fir.result %[[VAL_25]], %[[VAL_26]], %[[VAL_2]] : i32, f32, i1
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_11]], %[[VAL_12]], %[[VAL_13]] : i32, f32, i1
// CHECK:             }
// CHECK:             fir.result %[[VAL_27:.*]]#0, %[[VAL_27]]#1, %[[VAL_27]]#2 : i32, f32, i1
// CHECK:           }
// CHECK:           return %[[VAL_9]]#0 : i32
// CHECK:         }

func.func @test_1d_total_var(%input: !fir.box<!fir.array<?xf32>>, %mask: !hlfir.expr<?x!fir.logical<4>>) -> !hlfir.expr<1xi16> {
  %0 = hlfir.minloc %input mask %mask {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xf32>>, !hlfir.expr<?x!fir.logical<4>>) -> !hlfir.expr<1xi16>
  return %0 : !hlfir.expr<1xi16>
}
// CHECK-LABEL:   func.func @test_1d_total_var(
// CHECK-SAME:                                 %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?xf32>>,
// CHECK-SAME:                                 %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !hlfir.expr<?x!fir.logical<4>>) -> !hlfir.expr<1xi16> {
// CHECK:           %[[VAL_2:.*]] = arith.constant false
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant true
// CHECK:           %[[VAL_5:.*]] = arith.constant 3.40282347E+38 : f32
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i16
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_8:.*]] = fir.alloca !fir.array<1xi16>
// CHECK:           %[[VAL_9:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_7]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_10:.*]]:3 = fir.do_loop %[[VAL_11:.*]] = %[[VAL_3]] to %[[VAL_9]]#1 step %[[VAL_3]] iter_args(%[[VAL_12:.*]] = %[[VAL_6]], %[[VAL_13:.*]] = %[[VAL_5]], %[[VAL_14:.*]] = %[[VAL_4]]) -> (i16, f32, i1) {
// CHECK:             %[[VAL_15:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_11]] : (!hlfir.expr<?x!fir.logical<4>>, index) -> !fir.logical<4>
// CHECK:             %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (!fir.logical<4>) -> i1
// CHECK:             %[[VAL_17:.*]]:3 = fir.if %[[VAL_16]] -> (i16, f32, i1) {
// CHECK:               %[[VAL_18:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_7]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_19:.*]] = arith.subi %[[VAL_18]]#0, %[[VAL_3]] : index
// CHECK:               %[[VAL_20:.*]] = arith.addi %[[VAL_11]], %[[VAL_19]] : index
// CHECK:               %[[VAL_21:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_20]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:               %[[VAL_22:.*]] = fir.load %[[VAL_21]] : !fir.ref<f32>
// CHECK:               %[[VAL_23:.*]] = arith.cmpf olt, %[[VAL_22]], %[[VAL_13]] fastmath<contract> : f32
// CHECK:               %[[VAL_24:.*]] = arith.cmpf une, %[[VAL_13]], %[[VAL_13]] fastmath<contract> : f32
// CHECK:               %[[VAL_25:.*]] = arith.cmpf oeq, %[[VAL_22]], %[[VAL_22]] fastmath<contract> : f32
// CHECK:               %[[VAL_26:.*]] = arith.andi %[[VAL_24]], %[[VAL_25]] : i1
// CHECK:               %[[VAL_27:.*]] = arith.ori %[[VAL_23]], %[[VAL_26]] : i1
// CHECK:               %[[VAL_28:.*]] = arith.ori %[[VAL_27]], %[[VAL_14]] : i1
// CHECK:               %[[VAL_29:.*]] = fir.convert %[[VAL_11]] : (index) -> i16
// CHECK:               %[[VAL_30:.*]] = arith.select %[[VAL_28]], %[[VAL_29]], %[[VAL_12]] : i16
// CHECK:               %[[VAL_31:.*]] = arith.select %[[VAL_28]], %[[VAL_22]], %[[VAL_13]] : f32
// CHECK:               fir.result %[[VAL_30]], %[[VAL_31]], %[[VAL_2]] : i16, f32, i1
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_12]], %[[VAL_13]], %[[VAL_14]] : i16, f32, i1
// CHECK:             }
// CHECK:             fir.result %[[VAL_32:.*]]#0, %[[VAL_32]]#1, %[[VAL_32]]#2 : i16, f32, i1
// CHECK:           }
// CHECK:           %[[VAL_41:.*]] = hlfir.designate %[[VAL_8]] (%[[VAL_3]])  : (!fir.ref<!fir.array<1xi16>>, index) -> !fir.ref<i16>
// CHECK:           hlfir.assign %[[VAL_10]]#0 to %[[VAL_41]] : i16, !fir.ref<i16>
// CHECK:           %[[VAL_42:.*]] = hlfir.as_expr %[[VAL_8]] move %[[VAL_2]] : (!fir.ref<!fir.array<1xi16>>, i1) -> !hlfir.expr<1xi16>
// CHECK:           return %[[VAL_42]] : !hlfir.expr<1xi16>
// CHECK:         }

func.func @test_1d_dim_var(%input: !fir.box<!fir.array<?xf64>>, %mask: !hlfir.expr<?x!fir.logical<4>>) -> i64 {
  %dim = arith.constant 1 : i32
  %0 = hlfir.minloc %input dim %dim mask %mask {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xf64>>, i32, !hlfir.expr<?x!fir.logical<4>>) -> i64
  return %0 : i64
}
// CHECK-LABEL:   func.func @test_1d_dim_var(
// CHECK-SAME:                               %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?xf64>>,
// CHECK-SAME:                               %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !hlfir.expr<?x!fir.logical<4>>) -> i64 {
// CHECK:           %[[VAL_2:.*]] = arith.constant false
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant true
// CHECK:           %[[VAL_5:.*]] = arith.constant 1.7976931348623157E+308 : f64
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_8:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_7]] : (!fir.box<!fir.array<?xf64>>, index) -> (index, index, index)
// CHECK:           %[[VAL_9:.*]]:3 = fir.do_loop %[[VAL_10:.*]] = %[[VAL_3]] to %[[VAL_8]]#1 step %[[VAL_3]] iter_args(%[[VAL_11:.*]] = %[[VAL_6]], %[[VAL_12:.*]] = %[[VAL_5]], %[[VAL_13:.*]] = %[[VAL_4]]) -> (i64, f64, i1) {
// CHECK:             %[[VAL_14:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_10]] : (!hlfir.expr<?x!fir.logical<4>>, index) -> !fir.logical<4>
// CHECK:             %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (!fir.logical<4>) -> i1
// CHECK:             %[[VAL_16:.*]]:3 = fir.if %[[VAL_15]] -> (i64, f64, i1) {
// CHECK:               %[[VAL_17:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_7]] : (!fir.box<!fir.array<?xf64>>, index) -> (index, index, index)
// CHECK:               %[[VAL_18:.*]] = arith.subi %[[VAL_17]]#0, %[[VAL_3]] : index
// CHECK:               %[[VAL_19:.*]] = arith.addi %[[VAL_10]], %[[VAL_18]] : index
// CHECK:               %[[VAL_20:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_19]])  : (!fir.box<!fir.array<?xf64>>, index) -> !fir.ref<f64>
// CHECK:               %[[VAL_21:.*]] = fir.load %[[VAL_20]] : !fir.ref<f64>
// CHECK:               %[[VAL_22:.*]] = arith.cmpf olt, %[[VAL_21]], %[[VAL_12]] fastmath<contract> : f64
// CHECK:               %[[VAL_23:.*]] = arith.cmpf une, %[[VAL_12]], %[[VAL_12]] fastmath<contract> : f64
// CHECK:               %[[VAL_24:.*]] = arith.cmpf oeq, %[[VAL_21]], %[[VAL_21]] fastmath<contract> : f64
// CHECK:               %[[VAL_25:.*]] = arith.andi %[[VAL_23]], %[[VAL_24]] : i1
// CHECK:               %[[VAL_26:.*]] = arith.ori %[[VAL_22]], %[[VAL_25]] : i1
// CHECK:               %[[VAL_27:.*]] = arith.ori %[[VAL_26]], %[[VAL_13]] : i1
// CHECK:               %[[VAL_28:.*]] = fir.convert %[[VAL_10]] : (index) -> i64
// CHECK:               %[[VAL_29:.*]] = arith.select %[[VAL_27]], %[[VAL_28]], %[[VAL_11]] : i64
// CHECK:               %[[VAL_30:.*]] = arith.select %[[VAL_27]], %[[VAL_21]], %[[VAL_12]] : f64
// CHECK:               fir.result %[[VAL_29]], %[[VAL_30]], %[[VAL_2]] : i64, f64, i1
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_11]], %[[VAL_12]], %[[VAL_13]] : i64, f64, i1
// CHECK:             }
// CHECK:             fir.result %[[VAL_31:.*]]#0, %[[VAL_31]]#1, %[[VAL_31]]#2 : i64, f64, i1
// CHECK:           }
// CHECK:           return %[[VAL_9]]#0 : i64
// CHECK:         }

func.func @test_total_expr(%input: !hlfir.expr<?x?x?xf32>, %mask: !hlfir.expr<?x?x?x!fir.logical<4>>) -> !hlfir.expr<3xi32> {
  %0 = hlfir.minloc %input mask %mask {fastmath = #arith.fastmath<reassoc>} : (!hlfir.expr<?x?x?xf32>, !hlfir.expr<?x?x?x!fir.logical<4>>) -> !hlfir.expr<3xi32>
  return %0 : !hlfir.expr<3xi32>
}
// CHECK-LABEL:   func.func @test_total_expr(
// CHECK-SAME:                               %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !hlfir.expr<?x?x?xf32>,
// CHECK-SAME:                               %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !hlfir.expr<?x?x?x!fir.logical<4>>) -> !hlfir.expr<3xi32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant false
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant true
// CHECK:           %[[VAL_7:.*]] = arith.constant 3.40282347E+38 : f32
// CHECK:           %[[VAL_8:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_9:.*]] = fir.alloca !fir.array<3xi32>
// CHECK:           %[[VAL_10:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x?x?xf32>) -> !fir.shape<3>
// CHECK:           %[[VAL_11:.*]] = hlfir.get_extent %[[VAL_10]] {dim = 0 : index} : (!fir.shape<3>) -> index
// CHECK:           %[[VAL_12:.*]] = hlfir.get_extent %[[VAL_10]] {dim = 1 : index} : (!fir.shape<3>) -> index
// CHECK:           %[[VAL_13:.*]] = hlfir.get_extent %[[VAL_10]] {dim = 2 : index} : (!fir.shape<3>) -> index
// CHECK:           %[[VAL_14:.*]]:5 = fir.do_loop %[[VAL_15:.*]] = %[[VAL_5]] to %[[VAL_13]] step %[[VAL_5]] unordered iter_args(%[[VAL_16:.*]] = %[[VAL_8]], %[[VAL_17:.*]] = %[[VAL_8]], %[[VAL_18:.*]] = %[[VAL_8]], %[[VAL_19:.*]] = %[[VAL_7]], %[[VAL_20:.*]] = %[[VAL_6]]) -> (i32, i32, i32, f32, i1) {
// CHECK:             %[[VAL_21:.*]]:5 = fir.do_loop %[[VAL_22:.*]] = %[[VAL_5]] to %[[VAL_12]] step %[[VAL_5]] unordered iter_args(%[[VAL_23:.*]] = %[[VAL_16]], %[[VAL_24:.*]] = %[[VAL_17]], %[[VAL_25:.*]] = %[[VAL_18]], %[[VAL_26:.*]] = %[[VAL_19]], %[[VAL_27:.*]] = %[[VAL_20]]) -> (i32, i32, i32, f32, i1) {
// CHECK:               %[[VAL_28:.*]]:5 = fir.do_loop %[[VAL_29:.*]] = %[[VAL_5]] to %[[VAL_11]] step %[[VAL_5]] unordered iter_args(%[[VAL_30:.*]] = %[[VAL_23]], %[[VAL_31:.*]] = %[[VAL_24]], %[[VAL_32:.*]] = %[[VAL_25]], %[[VAL_33:.*]] = %[[VAL_26]], %[[VAL_34:.*]] = %[[VAL_27]]) -> (i32, i32, i32, f32, i1) {
// CHECK:                 %[[VAL_35:.*]] = hlfir.apply %[[VAL_1]], %[[VAL_29]], %[[VAL_22]], %[[VAL_15]] : (!hlfir.expr<?x?x?x!fir.logical<4>>, index, index, index) -> !fir.logical<4>
// CHECK:                 %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (!fir.logical<4>) -> i1
// CHECK:                 %[[VAL_37:.*]]:5 = fir.if %[[VAL_36]] -> (i32, i32, i32, f32, i1) {
// CHECK:                   %[[VAL_38:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_29]], %[[VAL_22]], %[[VAL_15]] : (!hlfir.expr<?x?x?xf32>, index, index, index) -> f32
// CHECK:                   %[[VAL_39:.*]] = arith.cmpf olt, %[[VAL_38]], %[[VAL_33]] fastmath<reassoc> : f32
// CHECK:                   %[[VAL_40:.*]] = arith.cmpf une, %[[VAL_33]], %[[VAL_33]] fastmath<reassoc> : f32
// CHECK:                   %[[VAL_41:.*]] = arith.cmpf oeq, %[[VAL_38]], %[[VAL_38]] fastmath<reassoc> : f32
// CHECK:                   %[[VAL_42:.*]] = arith.andi %[[VAL_40]], %[[VAL_41]] : i1
// CHECK:                   %[[VAL_43:.*]] = arith.ori %[[VAL_39]], %[[VAL_42]] : i1
// CHECK:                   %[[VAL_44:.*]] = arith.ori %[[VAL_43]], %[[VAL_34]] : i1
// CHECK:                   %[[VAL_45:.*]] = fir.convert %[[VAL_29]] : (index) -> i32
// CHECK:                   %[[VAL_46:.*]] = arith.select %[[VAL_44]], %[[VAL_45]], %[[VAL_30]] : i32
// CHECK:                   %[[VAL_47:.*]] = fir.convert %[[VAL_22]] : (index) -> i32
// CHECK:                   %[[VAL_48:.*]] = arith.select %[[VAL_44]], %[[VAL_47]], %[[VAL_31]] : i32
// CHECK:                   %[[VAL_49:.*]] = fir.convert %[[VAL_15]] : (index) -> i32
// CHECK:                   %[[VAL_50:.*]] = arith.select %[[VAL_44]], %[[VAL_49]], %[[VAL_32]] : i32
// CHECK:                   %[[VAL_51:.*]] = arith.select %[[VAL_44]], %[[VAL_38]], %[[VAL_33]] : f32
// CHECK:                   fir.result %[[VAL_46]], %[[VAL_48]], %[[VAL_50]], %[[VAL_51]], %[[VAL_4]] : i32, i32, i32, f32, i1
// CHECK:                 } else {
// CHECK:                   fir.result %[[VAL_30]], %[[VAL_31]], %[[VAL_32]], %[[VAL_33]], %[[VAL_34]] : i32, i32, i32, f32, i1
// CHECK:                 }
// CHECK:                 fir.result %[[VAL_52:.*]]#0, %[[VAL_52]]#1, %[[VAL_52]]#2, %[[VAL_52]]#3, %[[VAL_52]]#4 : i32, i32, i32, f32, i1
// CHECK:               }
// CHECK:               fir.result %[[VAL_53:.*]]#0, %[[VAL_53]]#1, %[[VAL_53]]#2, %[[VAL_53]]#3, %[[VAL_53]]#4 : i32, i32, i32, f32, i1
// CHECK:             }
// CHECK:             fir.result %[[VAL_54:.*]]#0, %[[VAL_54]]#1, %[[VAL_54]]#2, %[[VAL_54]]#3, %[[VAL_54]]#4 : i32, i32, i32, f32, i1
// CHECK:           }
// CHECK:           %[[VAL_58:.*]] = hlfir.designate %[[VAL_9]] (%[[VAL_5]])  : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
// CHECK:           hlfir.assign %[[VAL_14]]#0 to %[[VAL_58]] : i32, !fir.ref<i32>
// CHECK:           %[[VAL_61:.*]] = hlfir.designate %[[VAL_9]] (%[[VAL_3]])  : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
// CHECK:           hlfir.assign %[[VAL_14]]#1 to %[[VAL_61]] : i32, !fir.ref<i32>
// CHECK:           %[[VAL_64:.*]] = hlfir.designate %[[VAL_9]] (%[[VAL_2]])  : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
// CHECK:           hlfir.assign %[[VAL_14]]#2 to %[[VAL_64]] : i32, !fir.ref<i32>
// CHECK:           %[[VAL_65:.*]] = hlfir.as_expr %[[VAL_9]] move %[[VAL_4]] : (!fir.ref<!fir.array<3xi32>>, i1) -> !hlfir.expr<3xi32>
// CHECK:           return %[[VAL_65]] : !hlfir.expr<3xi32>
// CHECK:         }

func.func @test_partial_var(%input: !fir.box<!fir.array<?x?x?xf32>>, %mask: !fir.box<!fir.array<?x?x?x!fir.logical<4>>>) -> !hlfir.expr<?x?xi32> {
  %dim = arith.constant 2 : i32
  %0 = hlfir.minloc %input dim %dim mask %mask {fastmath = #arith.fastmath<reassoc>} : (!fir.box<!fir.array<?x?x?xf32>>, i32, !fir.box<!fir.array<?x?x?x!fir.logical<4>>>) -> !hlfir.expr<?x?xi32>
  return %0 : !hlfir.expr<?x?xi32>
}
// CHECK-LABEL:   func.func @test_partial_var(
// CHECK-SAME:                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?x?xf32>>,
// CHECK-SAME:                                %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?x?x!fir.logical<4>>>) -> !hlfir.expr<?x?xi32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant false
// CHECK:           %[[VAL_3:.*]] = arith.constant true
// CHECK:           %[[VAL_4:.*]] = arith.constant 3.40282347E+38 : f32
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_6:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_9:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_8]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_10:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_7]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_11:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_6]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_12:.*]] = fir.shape %[[VAL_9]]#1, %[[VAL_11]]#1 : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_13:.*]] = fir.is_present %[[VAL_1]] : (!fir.box<!fir.array<?x?x?x!fir.logical<4>>>) -> i1
// CHECK:           %[[VAL_14:.*]] = hlfir.elemental %[[VAL_12]] unordered : (!fir.shape<2>) -> !hlfir.expr<?x?xi32> {
// CHECK:           ^bb0(%[[VAL_15:.*]]: index, %[[VAL_16:.*]]: index):
// CHECK:             %[[VAL_17:.*]]:3 = fir.do_loop %[[VAL_18:.*]] = %[[VAL_7]] to %[[VAL_10]]#1 step %[[VAL_7]] unordered iter_args(%[[VAL_19:.*]] = %[[VAL_5]], %[[VAL_20:.*]] = %[[VAL_4]], %[[VAL_21:.*]] = %[[VAL_3]]) -> (i32, f32, i1) {
// CHECK:               %[[VAL_22:.*]] = fir.if %[[VAL_13]] -> (!fir.logical<4>) {
// CHECK:                 %[[VAL_23:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_8]] : (!fir.box<!fir.array<?x?x?x!fir.logical<4>>>, index) -> (index, index, index)
// CHECK:                 %[[VAL_24:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_7]] : (!fir.box<!fir.array<?x?x?x!fir.logical<4>>>, index) -> (index, index, index)
// CHECK:                 %[[VAL_25:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_6]] : (!fir.box<!fir.array<?x?x?x!fir.logical<4>>>, index) -> (index, index, index)
// CHECK:                 %[[VAL_26:.*]] = arith.subi %[[VAL_23]]#0, %[[VAL_7]] : index
// CHECK:                 %[[VAL_27:.*]] = arith.addi %[[VAL_15]], %[[VAL_26]] : index
// CHECK:                 %[[VAL_28:.*]] = arith.subi %[[VAL_24]]#0, %[[VAL_7]] : index
// CHECK:                 %[[VAL_29:.*]] = arith.addi %[[VAL_18]], %[[VAL_28]] : index
// CHECK:                 %[[VAL_30:.*]] = arith.subi %[[VAL_25]]#0, %[[VAL_7]] : index
// CHECK:                 %[[VAL_31:.*]] = arith.addi %[[VAL_16]], %[[VAL_30]] : index
// CHECK:                 %[[VAL_32:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_27]], %[[VAL_29]], %[[VAL_31]])  : (!fir.box<!fir.array<?x?x?x!fir.logical<4>>>, index, index, index) -> !fir.ref<!fir.logical<4>>
// CHECK:                 %[[VAL_33:.*]] = fir.load %[[VAL_32]] : !fir.ref<!fir.logical<4>>
// CHECK:                 fir.result %[[VAL_33]] : !fir.logical<4>
// CHECK:               } else {
// CHECK:                 %[[VAL_34:.*]] = fir.convert %[[VAL_3]] : (i1) -> !fir.logical<4>
// CHECK:                 fir.result %[[VAL_34]] : !fir.logical<4>
// CHECK:               }
// CHECK:               %[[VAL_35:.*]] = fir.convert %[[VAL_22]] : (!fir.logical<4>) -> i1
// CHECK:               %[[VAL_36:.*]]:3 = fir.if %[[VAL_35]] -> (i32, f32, i1) {
// CHECK:                 %[[VAL_37:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_8]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:                 %[[VAL_38:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_7]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:                 %[[VAL_39:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_6]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:                 %[[VAL_40:.*]] = arith.subi %[[VAL_37]]#0, %[[VAL_7]] : index
// CHECK:                 %[[VAL_41:.*]] = arith.addi %[[VAL_15]], %[[VAL_40]] : index
// CHECK:                 %[[VAL_42:.*]] = arith.subi %[[VAL_38]]#0, %[[VAL_7]] : index
// CHECK:                 %[[VAL_43:.*]] = arith.addi %[[VAL_18]], %[[VAL_42]] : index
// CHECK:                 %[[VAL_44:.*]] = arith.subi %[[VAL_39]]#0, %[[VAL_7]] : index
// CHECK:                 %[[VAL_45:.*]] = arith.addi %[[VAL_16]], %[[VAL_44]] : index
// CHECK:                 %[[VAL_46:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_41]], %[[VAL_43]], %[[VAL_45]])  : (!fir.box<!fir.array<?x?x?xf32>>, index, index, index) -> !fir.ref<f32>
// CHECK:                 %[[VAL_47:.*]] = fir.load %[[VAL_46]] : !fir.ref<f32>
// CHECK:                 %[[VAL_48:.*]] = arith.cmpf olt, %[[VAL_47]], %[[VAL_20]] fastmath<reassoc> : f32
// CHECK:                 %[[VAL_49:.*]] = arith.cmpf une, %[[VAL_20]], %[[VAL_20]] fastmath<reassoc> : f32
// CHECK:                 %[[VAL_50:.*]] = arith.cmpf oeq, %[[VAL_47]], %[[VAL_47]] fastmath<reassoc> : f32
// CHECK:                 %[[VAL_51:.*]] = arith.andi %[[VAL_49]], %[[VAL_50]] : i1
// CHECK:                 %[[VAL_52:.*]] = arith.ori %[[VAL_48]], %[[VAL_51]] : i1
// CHECK:                 %[[VAL_53:.*]] = arith.ori %[[VAL_52]], %[[VAL_21]] : i1
// CHECK:                 %[[VAL_54:.*]] = fir.convert %[[VAL_18]] : (index) -> i32
// CHECK:                 %[[VAL_55:.*]] = arith.select %[[VAL_53]], %[[VAL_54]], %[[VAL_19]] : i32
// CHECK:                 %[[VAL_56:.*]] = arith.select %[[VAL_53]], %[[VAL_47]], %[[VAL_20]] : f32
// CHECK:                 fir.result %[[VAL_55]], %[[VAL_56]], %[[VAL_2]] : i32, f32, i1
// CHECK:               } else {
// CHECK:                 fir.result %[[VAL_19]], %[[VAL_20]], %[[VAL_21]] : i32, f32, i1
// CHECK:               }
// CHECK:               fir.result %[[VAL_57:.*]]#0, %[[VAL_57]]#1, %[[VAL_57]]#2 : i32, f32, i1
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_17]]#0 : i32
// CHECK:           }
// CHECK:           return %[[VAL_14]] : !hlfir.expr<?x?xi32>
// CHECK:         }

func.func @test_total_expr_nomask(%input: !hlfir.expr<?x?x?xf32>) -> !hlfir.expr<3xi32> {
  %0 = hlfir.minloc %input {fastmath = #arith.fastmath<reassoc>} : (!hlfir.expr<?x?x?xf32>) -> !hlfir.expr<3xi32>
  return %0 : !hlfir.expr<3xi32>
}
// CHECK-LABEL:   func.func @test_total_expr_nomask(
// CHECK-SAME:                                      %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !hlfir.expr<?x?x?xf32>) -> !hlfir.expr<3xi32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant false
// CHECK:           %[[VAL_2:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 3.40282347E+38 : f32
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_9:.*]] = fir.alloca !fir.array<3xi32>
// CHECK:           %[[VAL_10:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x?x?xf32>) -> !fir.shape<3>
// CHECK:           %[[VAL_11:.*]] = hlfir.get_extent %[[VAL_10]] {dim = 0 : index} : (!fir.shape<3>) -> index
// CHECK:           %[[VAL_12:.*]] = hlfir.get_extent %[[VAL_10]] {dim = 1 : index} : (!fir.shape<3>) -> index
// CHECK:           %[[VAL_13:.*]] = hlfir.get_extent %[[VAL_10]] {dim = 2 : index} : (!fir.shape<3>) -> index
// CHECK:           %[[VAL_14:.*]] = arith.cmpi ne, %[[VAL_11]], %[[VAL_8]] : index
// CHECK:           %[[VAL_15:.*]] = arith.cmpi ne, %[[VAL_12]], %[[VAL_8]] : index
// CHECK:           %[[VAL_16:.*]] = arith.andi %[[VAL_14]], %[[VAL_15]] : i1
// CHECK:           %[[VAL_17:.*]] = arith.cmpi ne, %[[VAL_13]], %[[VAL_8]] : index
// CHECK:           %[[VAL_18:.*]] = arith.andi %[[VAL_16]], %[[VAL_17]] : i1
// CHECK:           %[[VAL_19:.*]]:4 = fir.if %[[VAL_18]] -> (i32, i32, i32, f32) {
// CHECK:             %[[VAL_20:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_7]], %[[VAL_7]], %[[VAL_7]] : (!hlfir.expr<?x?x?xf32>, index, index, index) -> f32
// CHECK:             fir.result %[[VAL_6]], %[[VAL_6]], %[[VAL_6]], %[[VAL_20]] : i32, i32, i32, f32
// CHECK:           } else {
// CHECK:             fir.result %[[VAL_5]], %[[VAL_5]], %[[VAL_5]], %[[VAL_4]] : i32, i32, i32, f32
// CHECK:           }
// CHECK:           %[[VAL_21:.*]]:4 = fir.do_loop %[[VAL_22:.*]] = %[[VAL_7]] to %[[VAL_13]] step %[[VAL_7]] unordered iter_args(%[[VAL_23:.*]] = %[[VAL_24:.*]]#0, %[[VAL_25:.*]] = %[[VAL_24]]#1, %[[VAL_26:.*]] = %[[VAL_24]]#2, %[[VAL_27:.*]] = %[[VAL_24]]#3) -> (i32, i32, i32, f32) {
// CHECK:             %[[VAL_28:.*]]:4 = fir.do_loop %[[VAL_29:.*]] = %[[VAL_7]] to %[[VAL_12]] step %[[VAL_7]] unordered iter_args(%[[VAL_30:.*]] = %[[VAL_23]], %[[VAL_31:.*]] = %[[VAL_25]], %[[VAL_32:.*]] = %[[VAL_26]], %[[VAL_33:.*]] = %[[VAL_27]]) -> (i32, i32, i32, f32) {
// CHECK:               %[[VAL_34:.*]]:4 = fir.do_loop %[[VAL_35:.*]] = %[[VAL_7]] to %[[VAL_11]] step %[[VAL_7]] unordered iter_args(%[[VAL_36:.*]] = %[[VAL_30]], %[[VAL_37:.*]] = %[[VAL_31]], %[[VAL_38:.*]] = %[[VAL_32]], %[[VAL_39:.*]] = %[[VAL_33]]) -> (i32, i32, i32, f32) {
// CHECK:                 %[[VAL_40:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_35]], %[[VAL_29]], %[[VAL_22]] : (!hlfir.expr<?x?x?xf32>, index, index, index) -> f32
// CHECK:                 %[[VAL_41:.*]] = arith.cmpf olt, %[[VAL_40]], %[[VAL_39]] fastmath<reassoc> : f32
// CHECK:                 %[[VAL_42:.*]] = arith.cmpf une, %[[VAL_39]], %[[VAL_39]] fastmath<reassoc> : f32
// CHECK:                 %[[VAL_43:.*]] = arith.cmpf oeq, %[[VAL_40]], %[[VAL_40]] fastmath<reassoc> : f32
// CHECK:                 %[[VAL_44:.*]] = arith.andi %[[VAL_42]], %[[VAL_43]] : i1
// CHECK:                 %[[VAL_45:.*]] = arith.ori %[[VAL_41]], %[[VAL_44]] : i1
// CHECK:                 %[[VAL_46:.*]] = fir.convert %[[VAL_35]] : (index) -> i32
// CHECK:                 %[[VAL_47:.*]] = arith.select %[[VAL_45]], %[[VAL_46]], %[[VAL_36]] : i32
// CHECK:                 %[[VAL_48:.*]] = fir.convert %[[VAL_29]] : (index) -> i32
// CHECK:                 %[[VAL_49:.*]] = arith.select %[[VAL_45]], %[[VAL_48]], %[[VAL_37]] : i32
// CHECK:                 %[[VAL_50:.*]] = fir.convert %[[VAL_22]] : (index) -> i32
// CHECK:                 %[[VAL_51:.*]] = arith.select %[[VAL_45]], %[[VAL_50]], %[[VAL_38]] : i32
// CHECK:                 %[[VAL_52:.*]] = arith.select %[[VAL_45]], %[[VAL_40]], %[[VAL_39]] : f32
// CHECK:                 fir.result %[[VAL_47]], %[[VAL_49]], %[[VAL_51]], %[[VAL_52]] : i32, i32, i32, f32
// CHECK:               }
// CHECK:               fir.result %[[VAL_53:.*]]#0, %[[VAL_53]]#1, %[[VAL_53]]#2, %[[VAL_53]]#3 : i32, i32, i32, f32
// CHECK:             }
// CHECK:             fir.result %[[VAL_54:.*]]#0, %[[VAL_54]]#1, %[[VAL_54]]#2, %[[VAL_54]]#3 : i32, i32, i32, f32
// CHECK:           }
// CHECK:           %[[VAL_55:.*]] = hlfir.designate %[[VAL_9]] (%[[VAL_7]])  : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
// CHECK:           hlfir.assign %[[VAL_56:.*]]#0 to %[[VAL_55]] : i32, !fir.ref<i32>
// CHECK:           %[[VAL_57:.*]] = hlfir.designate %[[VAL_9]] (%[[VAL_3]])  : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
// CHECK:           hlfir.assign %[[VAL_56]]#1 to %[[VAL_57]] : i32, !fir.ref<i32>
// CHECK:           %[[VAL_58:.*]] = hlfir.designate %[[VAL_9]] (%[[VAL_2]])  : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
// CHECK:           hlfir.assign %[[VAL_56]]#2 to %[[VAL_58]] : i32, !fir.ref<i32>
// CHECK:           %[[VAL_59:.*]] = hlfir.as_expr %[[VAL_9]] move %[[VAL_1]] : (!fir.ref<!fir.array<3xi32>>, i1) -> !hlfir.expr<3xi32>
// CHECK:           return %[[VAL_59]] : !hlfir.expr<3xi32>
// CHECK:         }

func.func @test_partial_var_nomask(%input: !fir.box<!fir.array<?x?x?xf32>>) -> !hlfir.expr<?x?xi32> {
  %dim = arith.constant 2 : i32
  %0 = hlfir.minloc %input dim %dim {fastmath = #arith.fastmath<reassoc>} : (!fir.box<!fir.array<?x?x?xf32>>, i32) -> !hlfir.expr<?x?xi32>
  return %0 : !hlfir.expr<?x?xi32>
}
// CHECK-LABEL:   func.func @test_partial_var_nomask(
// CHECK-SAME:                                       %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x?x?xf32>>) -> !hlfir.expr<?x?xi32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 3.40282347E+38 : f32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_7:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_6]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_8:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_9:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_10:.*]] = fir.shape %[[VAL_7]]#1, %[[VAL_9]]#1 : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_11:.*]] = hlfir.elemental %[[VAL_10]] unordered : (!fir.shape<2>) -> !hlfir.expr<?x?xi32> {
// CHECK:           ^bb0(%[[VAL_12:.*]]: index, %[[VAL_13:.*]]: index):
// CHECK:             %[[VAL_14:.*]] = arith.cmpi ne, %[[VAL_8]]#1, %[[VAL_6]] : index
// CHECK:             %[[VAL_15:.*]]:2 = fir.if %[[VAL_14]] -> (i32, f32) {
// CHECK:               %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_6]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_17:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_18:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_19:.*]] = arith.subi %[[VAL_16]]#0, %[[VAL_5]] : index
// CHECK:               %[[VAL_20:.*]] = arith.addi %[[VAL_12]], %[[VAL_19]] : index
// CHECK:               %[[VAL_21:.*]] = arith.subi %[[VAL_18]]#0, %[[VAL_5]] : index
// CHECK:               %[[VAL_22:.*]] = arith.addi %[[VAL_13]], %[[VAL_21]] : index
// CHECK:               %[[VAL_23:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_20]], %[[VAL_17]]#0, %[[VAL_22]])  : (!fir.box<!fir.array<?x?x?xf32>>, index, index, index) -> !fir.ref<f32>
// CHECK:               %[[VAL_24:.*]] = fir.load %[[VAL_23]] : !fir.ref<f32>
// CHECK:               fir.result %[[VAL_3]], %[[VAL_24]] : i32, f32
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_2]], %[[VAL_1]] : i32, f32
// CHECK:             }
// CHECK:             %[[VAL_25:.*]]:2 = fir.do_loop %[[VAL_26:.*]] = %[[VAL_5]] to %[[VAL_8]]#1 step %[[VAL_5]] unordered iter_args(%[[VAL_27:.*]] = %[[VAL_28:.*]]#0, %[[VAL_29:.*]] = %[[VAL_28]]#1) -> (i32, f32) {
// CHECK:               %[[VAL_30:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_6]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_31:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_32:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_33:.*]] = arith.subi %[[VAL_30]]#0, %[[VAL_5]] : index
// CHECK:               %[[VAL_34:.*]] = arith.addi %[[VAL_12]], %[[VAL_33]] : index
// CHECK:               %[[VAL_35:.*]] = arith.subi %[[VAL_31]]#0, %[[VAL_5]] : index
// CHECK:               %[[VAL_36:.*]] = arith.addi %[[VAL_26]], %[[VAL_35]] : index
// CHECK:               %[[VAL_37:.*]] = arith.subi %[[VAL_32]]#0, %[[VAL_5]] : index
// CHECK:               %[[VAL_38:.*]] = arith.addi %[[VAL_13]], %[[VAL_37]] : index
// CHECK:               %[[VAL_39:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_34]], %[[VAL_36]], %[[VAL_38]])  : (!fir.box<!fir.array<?x?x?xf32>>, index, index, index) -> !fir.ref<f32>
// CHECK:               %[[VAL_40:.*]] = fir.load %[[VAL_39]] : !fir.ref<f32>
// CHECK:               %[[VAL_41:.*]] = arith.cmpf olt, %[[VAL_40]], %[[VAL_29]] fastmath<reassoc> : f32
// CHECK:               %[[VAL_42:.*]] = arith.cmpf une, %[[VAL_29]], %[[VAL_29]] fastmath<reassoc> : f32
// CHECK:               %[[VAL_43:.*]] = arith.cmpf oeq, %[[VAL_40]], %[[VAL_40]] fastmath<reassoc> : f32
// CHECK:               %[[VAL_44:.*]] = arith.andi %[[VAL_42]], %[[VAL_43]] : i1
// CHECK:               %[[VAL_45:.*]] = arith.ori %[[VAL_41]], %[[VAL_44]] : i1
// CHECK:               %[[VAL_46:.*]] = fir.convert %[[VAL_26]] : (index) -> i32
// CHECK:               %[[VAL_47:.*]] = arith.select %[[VAL_45]], %[[VAL_46]], %[[VAL_27]] : i32
// CHECK:               %[[VAL_48:.*]] = arith.select %[[VAL_45]], %[[VAL_40]], %[[VAL_29]] : f32
// CHECK:               fir.result %[[VAL_47]], %[[VAL_48]] : i32, f32
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_49:.*]]#0 : i32
// CHECK:           }
// CHECK:           return %[[VAL_11]] : !hlfir.expr<?x?xi32>
// CHECK:         }

// Test that 'nnan' allows using LARGEST value as the reduction init.
func.func @test_total_expr_nnan(%input: !hlfir.expr<?x?x?xf32>) -> !hlfir.expr<3xi32> {
  %0 = hlfir.minloc %input {fastmath = #arith.fastmath<nnan>} : (!hlfir.expr<?x?x?xf32>) -> !hlfir.expr<3xi32>
  return %0 : !hlfir.expr<3xi32>
}
// CHECK-LABEL:   func.func @test_total_expr_nnan(
// CHECK-SAME:                                    %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !hlfir.expr<?x?x?xf32>) -> !hlfir.expr<3xi32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant false
// CHECK:           %[[VAL_2:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 3.40282347E+38 : f32
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_7:.*]] = fir.alloca !fir.array<3xi32>
// CHECK:           %[[VAL_8:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x?x?xf32>) -> !fir.shape<3>
// CHECK:           %[[VAL_9:.*]] = hlfir.get_extent %[[VAL_8]] {dim = 0 : index} : (!fir.shape<3>) -> index
// CHECK:           %[[VAL_10:.*]] = hlfir.get_extent %[[VAL_8]] {dim = 1 : index} : (!fir.shape<3>) -> index
// CHECK:           %[[VAL_11:.*]] = hlfir.get_extent %[[VAL_8]] {dim = 2 : index} : (!fir.shape<3>) -> index
// CHECK:           %[[VAL_12:.*]]:4 = fir.do_loop %[[VAL_13:.*]] = %[[VAL_4]] to %[[VAL_11]] step %[[VAL_4]] iter_args(%[[VAL_14:.*]] = %[[VAL_6]], %[[VAL_15:.*]] = %[[VAL_6]], %[[VAL_16:.*]] = %[[VAL_6]], %[[VAL_17:.*]] = %[[VAL_5]]) -> (i32, i32, i32, f32) {
// CHECK:             %[[VAL_18:.*]]:4 = fir.do_loop %[[VAL_19:.*]] = %[[VAL_4]] to %[[VAL_10]] step %[[VAL_4]] iter_args(%[[VAL_20:.*]] = %[[VAL_14]], %[[VAL_21:.*]] = %[[VAL_15]], %[[VAL_22:.*]] = %[[VAL_16]], %[[VAL_23:.*]] = %[[VAL_17]]) -> (i32, i32, i32, f32) {
// CHECK:               %[[VAL_24:.*]]:4 = fir.do_loop %[[VAL_25:.*]] = %[[VAL_4]] to %[[VAL_9]] step %[[VAL_4]] iter_args(%[[VAL_26:.*]] = %[[VAL_20]], %[[VAL_27:.*]] = %[[VAL_21]], %[[VAL_28:.*]] = %[[VAL_22]], %[[VAL_29:.*]] = %[[VAL_23]]) -> (i32, i32, i32, f32) {
// CHECK:                 %[[VAL_30:.*]] = hlfir.apply %[[VAL_0]], %[[VAL_25]], %[[VAL_19]], %[[VAL_13]] : (!hlfir.expr<?x?x?xf32>, index, index, index) -> f32
// CHECK:                 %[[VAL_31:.*]] = arith.cmpf olt, %[[VAL_30]], %[[VAL_29]] fastmath<nnan> : f32
// CHECK:                 %[[VAL_32:.*]] = arith.cmpf une, %[[VAL_29]], %[[VAL_29]] fastmath<nnan> : f32
// CHECK:                 %[[VAL_33:.*]] = arith.cmpf oeq, %[[VAL_30]], %[[VAL_30]] fastmath<nnan> : f32
// CHECK:                 %[[VAL_34:.*]] = arith.andi %[[VAL_32]], %[[VAL_33]] : i1
// CHECK:                 %[[VAL_35:.*]] = arith.ori %[[VAL_31]], %[[VAL_34]] : i1
// CHECK:                 %[[VAL_36:.*]] = fir.convert %[[VAL_25]] : (index) -> i32
// CHECK:                 %[[VAL_37:.*]] = arith.select %[[VAL_35]], %[[VAL_36]], %[[VAL_26]] : i32
// CHECK:                 %[[VAL_38:.*]] = fir.convert %[[VAL_19]] : (index) -> i32
// CHECK:                 %[[VAL_39:.*]] = arith.select %[[VAL_35]], %[[VAL_38]], %[[VAL_27]] : i32
// CHECK:                 %[[VAL_40:.*]] = fir.convert %[[VAL_13]] : (index) -> i32
// CHECK:                 %[[VAL_41:.*]] = arith.select %[[VAL_35]], %[[VAL_40]], %[[VAL_28]] : i32
// CHECK:                 %[[VAL_42:.*]] = arith.select %[[VAL_35]], %[[VAL_30]], %[[VAL_29]] : f32
// CHECK:                 fir.result %[[VAL_37]], %[[VAL_39]], %[[VAL_41]], %[[VAL_42]] : i32, i32, i32, f32
// CHECK:               }
// CHECK:               fir.result %[[VAL_43:.*]]#0, %[[VAL_43]]#1, %[[VAL_43]]#2, %[[VAL_43]]#3 : i32, i32, i32, f32
// CHECK:             }
// CHECK:             fir.result %[[VAL_44:.*]]#0, %[[VAL_44]]#1, %[[VAL_44]]#2, %[[VAL_44]]#3 : i32, i32, i32, f32
// CHECK:           }
// CHECK:           %[[VAL_45:.*]] = hlfir.designate %[[VAL_7]] (%[[VAL_4]])  : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
// CHECK:           hlfir.assign %[[VAL_46:.*]]#0 to %[[VAL_45]] : i32, !fir.ref<i32>
// CHECK:           %[[VAL_47:.*]] = hlfir.designate %[[VAL_7]] (%[[VAL_3]])  : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
// CHECK:           hlfir.assign %[[VAL_46]]#1 to %[[VAL_47]] : i32, !fir.ref<i32>
// CHECK:           %[[VAL_48:.*]] = hlfir.designate %[[VAL_7]] (%[[VAL_2]])  : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
// CHECK:           hlfir.assign %[[VAL_46]]#2 to %[[VAL_48]] : i32, !fir.ref<i32>
// CHECK:           %[[VAL_49:.*]] = hlfir.as_expr %[[VAL_7]] move %[[VAL_1]] : (!fir.ref<!fir.array<3xi32>>, i1) -> !hlfir.expr<3xi32>
// CHECK:           return %[[VAL_49]] : !hlfir.expr<3xi32>
// CHECK:         }

// Character comparisons are not supported yet.
func.func @test_character(%input: !fir.box<!fir.array<?x!fir.char<1>>>) -> !hlfir.expr<1xi32> {
  %0 = hlfir.minloc %input : (!fir.box<!fir.array<?x!fir.char<1>>>) -> !hlfir.expr<1xi32>
  return %0 : !hlfir.expr<1xi32>
}
// CHECK-LABEL:   func.func @test_character(
// CHECK:           hlfir.minloc

// BACK is not supported yet.
func.func @test_back(%input: !hlfir.expr<?xi32>) -> !hlfir.expr<1xi32> {
  %back = arith.constant true
  %0 = hlfir.minloc %input back %back : (!hlfir.expr<?xi32>, i1) -> !hlfir.expr<1xi32>
  return %0 : !hlfir.expr<1xi32>
}
// CHECK-LABEL:   func.func @test_back(
// CHECK:           hlfir.minloc
