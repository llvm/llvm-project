// Test nested hlfir.elemental code generation
// RUN: fir-opt %s --bufferize-hlfir | FileCheck %s

// CHECK-LABEL:   func.func @_QPtest(
// CHECK-SAME:                       %[[VAL_0:.*]]: !fir.ref<f32> {fir.bindc_name = "pi"},
// CHECK-SAME:                       %[[VAL_1:.*]]: !fir.ref<!fir.array<2xf32>> {fir.bindc_name = "h1"}) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_3]]) {uniq_name = "_QFtestEh1"} : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xf32>>, !fir.ref<!fir.array<2xf32>>)
// CHECK:           %[[VAL_5:.*]] = fir.alloca i32 {bindc_name = "k", uniq_name = "_QFtestEk"}
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] {uniq_name = "_QFtestEk"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_7:.*]] = fir.alloca i32 {bindc_name = "l", uniq_name = "_QFtestEl"}
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] {uniq_name = "_QFtestEl"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_9:.*]] = fir.address_of(@_QFtestECn) : !fir.ref<i32>
// CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_9]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFtestECn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtestEpi"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
// CHECK:           %[[VAL_12:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_13:.*]] = fir.allocmem !fir.array<2xf32> {bindc_name = ".tmp.array", uniq_name = ""}
// CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_13]](%[[VAL_12]]) {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<2xf32>>, !fir.shape<1>) -> (!fir.heap<!fir.array<2xf32>>, !fir.heap<!fir.array<2xf32>>)
// CHECK:           %[[VAL_15:.*]] = arith.constant true
// CHECK:           %[[VAL_16:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_17:.*]] = %[[VAL_16]] to %[[VAL_2]] step %[[VAL_16]] {
// CHECK:             %[[VAL_18:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:             %[[VAL_19:.*]] = fir.allocmem !fir.array<2xf32> {bindc_name = ".tmp.array", uniq_name = ""}
// CHECK:             %[[VAL_20:.*]]:2 = hlfir.declare %[[VAL_19]](%[[VAL_18]]) {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<2xf32>>, !fir.shape<1>) -> (!fir.heap<!fir.array<2xf32>>, !fir.heap<!fir.array<2xf32>>)
// CHECK:             %[[VAL_21:.*]] = arith.constant true
// CHECK:             %[[VAL_22:.*]] = arith.constant 1 : index
// CHECK:             fir.do_loop %[[VAL_23:.*]] = %[[VAL_22]] to %[[VAL_2]] step %[[VAL_22]] {
// CHECK:               %[[VAL_24:.*]] = fir.load %[[VAL_11]]#0 : !fir.ref<f32>
// CHECK:               %[[VAL_25:.*]] = hlfir.designate %[[VAL_20]]#0 (%[[VAL_23]])  : (!fir.heap<!fir.array<2xf32>>, index) -> !fir.ref<f32>
// CHECK:               hlfir.assign %[[VAL_24]] to %[[VAL_25]] temporary_lhs : f32, !fir.ref<f32>
// CHECK:             }
// CHECK:             %[[VAL_26:.*]] = fir.undefined tuple<!fir.heap<!fir.array<2xf32>>, i1>
// CHECK:             %[[VAL_27:.*]] = fir.insert_value %[[VAL_26]], %[[VAL_21]], [1 : index] : (tuple<!fir.heap<!fir.array<2xf32>>, i1>, i1) -> tuple<!fir.heap<!fir.array<2xf32>>, i1>
// CHECK:             %[[VAL_28:.*]] = fir.insert_value %[[VAL_27]], %[[VAL_20]]#0, [0 : index] : (tuple<!fir.heap<!fir.array<2xf32>>, i1>, !fir.heap<!fir.array<2xf32>>) -> tuple<!fir.heap<!fir.array<2xf32>>, i1>
// CHECK:             %[[VAL_29:.*]] = fir.convert %[[VAL_20]]#0 : (!fir.heap<!fir.array<2xf32>>) -> !fir.ref<!fir.array<2xf32>>
// CHECK:             %[[VAL_30:.*]] = fir.convert %[[VAL_20]]#1 : (!fir.heap<!fir.array<2xf32>>) -> !fir.ref<!fir.array<2xf32>>
// CHECK:             %[[VAL_31:.*]] = fir.embox %[[VAL_29]](%[[VAL_18]]) : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xf32>>
// CHECK:             %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (!fir.box<!fir.array<2xf32>>) -> !fir.box<!fir.array<?xf32>>
// CHECK:             %[[VAL_33:.*]] = fir.call @_QPfoo(%[[VAL_32]]) fastmath<contract> : (!fir.box<!fir.array<?xf32>>) -> f32
// CHECK:             %[[VAL_34:.*]] = fir.convert %[[VAL_30]] : (!fir.ref<!fir.array<2xf32>>) -> !fir.heap<!fir.array<2xf32>>
// CHECK:             fir.freemem %[[VAL_34]] : !fir.heap<!fir.array<2xf32>>
// CHECK:             %[[VAL_35:.*]] = hlfir.designate %[[VAL_14]]#0 (%[[VAL_17]])  : (!fir.heap<!fir.array<2xf32>>, index) -> !fir.ref<f32>
// CHECK:             hlfir.assign %[[VAL_33]] to %[[VAL_35]] temporary_lhs : f32, !fir.ref<f32>
// CHECK:           }
// CHECK:           %[[VAL_36:.*]] = fir.undefined tuple<!fir.heap<!fir.array<2xf32>>, i1>
// CHECK:           %[[VAL_37:.*]] = fir.insert_value %[[VAL_36]], %[[VAL_15]], [1 : index] : (tuple<!fir.heap<!fir.array<2xf32>>, i1>, i1) -> tuple<!fir.heap<!fir.array<2xf32>>, i1>
// CHECK:           %[[VAL_38:.*]] = fir.insert_value %[[VAL_37]], %[[VAL_14]]#0, [0 : index] : (tuple<!fir.heap<!fir.array<2xf32>>, i1>, !fir.heap<!fir.array<2xf32>>) -> tuple<!fir.heap<!fir.array<2xf32>>, i1>
// CHECK:           hlfir.assign %[[VAL_14]]#0 to %[[VAL_4]]#0 : !fir.heap<!fir.array<2xf32>>, !fir.ref<!fir.array<2xf32>>
// CHECK:           fir.freemem %[[VAL_14]]#1 : !fir.heap<!fir.array<2xf32>>
// CHECK:           return
// CHECK:         }
func.func @_QPtest(%arg0: !fir.ref<f32> {fir.bindc_name = "pi"}, %arg1: !fir.ref<!fir.array<2xf32>> {fir.bindc_name = "h1"}) {
  %c2 = arith.constant 2 : index
  %0 = fir.shape %c2 : (index) -> !fir.shape<1>
  %1:2 = hlfir.declare %arg1(%0) {uniq_name = "_QFtestEh1"} : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xf32>>, !fir.ref<!fir.array<2xf32>>)
  %2 = fir.alloca i32 {bindc_name = "k", uniq_name = "_QFtestEk"}
  %3:2 = hlfir.declare %2 {uniq_name = "_QFtestEk"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4 = fir.alloca i32 {bindc_name = "l", uniq_name = "_QFtestEl"}
  %5:2 = hlfir.declare %4 {uniq_name = "_QFtestEl"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %6 = fir.address_of(@_QFtestECn) : !fir.ref<i32>
  %7:2 = hlfir.declare %6 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFtestECn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %8:2 = hlfir.declare %arg0 {uniq_name = "_QFtestEpi"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %9 = fir.shape %c2 : (index) -> !fir.shape<1>
  %10 = hlfir.elemental %9 : (!fir.shape<1>) -> !hlfir.expr<2xf32> {
  ^bb0(%arg2: index):
    %11 = fir.shape %c2 : (index) -> !fir.shape<1>
    %12 = hlfir.elemental %11 : (!fir.shape<1>) -> !hlfir.expr<2xf32> {
    ^bb0(%arg3: index):
      %17 = fir.load %8#0 : !fir.ref<f32>
      hlfir.yield_element %17 : f32
    }
    %13:3 = hlfir.associate %12(%11) {uniq_name = "adapt.valuebyref"} : (!hlfir.expr<2xf32>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xf32>>, !fir.ref<!fir.array<2xf32>>, i1)
    %14 = fir.embox %13#0(%11) : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xf32>>
    %15 = fir.convert %14 : (!fir.box<!fir.array<2xf32>>) -> !fir.box<!fir.array<?xf32>>
    %16 = fir.call @_QPfoo(%15) fastmath<contract> : (!fir.box<!fir.array<?xf32>>) -> f32
    hlfir.end_associate %13#1, %13#2 : !fir.ref<!fir.array<2xf32>>, i1
    hlfir.destroy %12 : !hlfir.expr<2xf32>
    hlfir.yield_element %16 : f32
  }
  hlfir.assign %10 to %1#0 : !hlfir.expr<2xf32>, !fir.ref<!fir.array<2xf32>>
  hlfir.destroy %10 : !hlfir.expr<2xf32>
  return
}
fir.global internal @_QFtestECn constant : i32 {
  %c2_i32 = arith.constant 2 : i32
  fir.has_value %c2_i32 : i32
}
func.func private @_QPfoo(!fir.box<!fir.array<?xf32>>) -> f32
