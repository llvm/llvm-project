// Tests converting `fir.local` ops that have `dealloc` regions.

// RUN: fir-opt --split-input-file --simplify-fir-operations %s | FileCheck %s

fir.local {type = local} @_QFlocalizer_with_dealloc_regionEa_private_box_Uxi32 : !fir.box<!fir.array<?xi32>> init {
^bb0(%arg0: !fir.ref<!fir.box<!fir.array<?xi32>>>, %arg1: !fir.ref<!fir.box<!fir.array<?xi32>>>):
  %c0 = arith.constant 0 : index
  %0 = fir.load %arg0 : !fir.ref<!fir.box<!fir.array<?xi32>>>
  %1:3 = fir.box_dims %0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  %2 = fir.shape %1#1 : (index) -> !fir.shape<1>
  %3 = fir.allocmem !fir.array<?xi32>, %1#1 {bindc_name = ".tmp", uniq_name = ""}
  %4 = fir.declare %3(%2) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.heap<!fir.array<?xi32>>
  %5 = fir.embox %4(%2) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
  %6 = fir.shape_shift %1#0, %1#1 : (index, index) -> !fir.shapeshift<1>
  %7 = fir.rebox %5(%6) : (!fir.box<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?xi32>>
  fir.store %7 to %arg1 : !fir.ref<!fir.box<!fir.array<?xi32>>>
  fir.yield(%arg1 : !fir.ref<!fir.box<!fir.array<?xi32>>>)
} dealloc {
^bb0(%arg0: !fir.ref<!fir.box<!fir.array<?xi32>>>):
  %c0_i64 = arith.constant 0 : i64
  %0 = fir.load %arg0 : !fir.ref<!fir.box<!fir.array<?xi32>>>
  %1 = fir.box_addr %0 : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
  %2 = fir.convert %1 : (!fir.ref<!fir.array<?xi32>>) -> i64
  %3 = arith.cmpi ne, %2, %c0_i64 : i64
  fir.if %3 {
    %4 = fir.convert %1 : (!fir.ref<!fir.array<?xi32>>) -> !fir.heap<!fir.array<?xi32>>
    fir.freemem %4 : !fir.heap<!fir.array<?xi32>>
  }
  fir.yield
}

func.func @_QPlocalizer_with_dealloc_region(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c42_i32 = arith.constant 42 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.alloca !fir.box<!fir.array<?xi32>>
  %1 = fir.dummy_scope : !fir.dscope
  %2 = fir.declare %arg0 dummy_scope %1 {uniq_name = "_QFlocalizer_with_dealloc_regionEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %3 = fir.load %2 : !fir.ref<i32>
  %4 = fir.convert %3 : (i32) -> index
  %5 = arith.cmpi sgt, %4, %c0 : index
  %6 = arith.select %5, %4, %c0 : index
  %7 = fir.alloca !fir.array<?xi32>, %6 {bindc_name = "a", uniq_name = "_QFlocalizer_with_dealloc_regionEa"}
  %8 = fir.shape %6 : (index) -> !fir.shape<1>
  %9 = fir.declare %7(%8) {uniq_name = "_QFlocalizer_with_dealloc_regionEa"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<?xi32>>
  %10 = fir.embox %9(%8) : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
  fir.store %10 to %0 : !fir.ref<!fir.box<!fir.array<?xi32>>>
  fir.do_concurrent {
    %11 = fir.alloca i32 {bindc_name = "i"}
    %12 = fir.declare %11 {uniq_name = "_QFlocalizer_with_dealloc_regionEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    fir.do_concurrent.loop (%arg1) = (%c1) to (%4) step (%c1) local(@_QFlocalizer_with_dealloc_regionEa_private_box_Uxi32 %0 -> %arg2 : !fir.ref<!fir.box<!fir.array<?xi32>>>) {
      %13 = fir.convert %arg1 : (index) -> i32
      fir.store %13 to %12 : !fir.ref<i32>
      %14 = fir.declare %arg2 {uniq_name = "_QFlocalizer_with_dealloc_regionEa"} : (!fir.ref<!fir.box<!fir.array<?xi32>>>) -> !fir.ref<!fir.box<!fir.array<?xi32>>>
      %15 = fir.load %14 : !fir.ref<!fir.box<!fir.array<?xi32>>>
      %16 = fir.load %12 : !fir.ref<i32>
      %17 = fir.convert %16 : (i32) -> i64
      %18:3 = fir.box_dims %15, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
      %19 = fir.shift %18#0 : (index) -> !fir.shift<1>
      %20 = fir.array_coor %15(%19) %17 : (!fir.box<!fir.array<?xi32>>, !fir.shift<1>, i64) -> !fir.ref<i32>
      fir.store %c42_i32 to %20 : !fir.ref<i32>
    }
  }
  return
}

// CHECK-LABEL:   func.func @_QPlocalizer_with_dealloc_region(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 42 : i32
// CHECK:           %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "i"}
// CHECK:           %[[VAL_5:.*]] = fir.declare %[[VAL_4]] {uniq_name = "_QFlocalizer_with_dealloc_regionEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_6:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
// CHECK:           %[[VAL_7:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_8:.*]] = fir.declare %[[ARG0]] dummy_scope %[[VAL_7]] {uniq_name = "_QFlocalizer_with_dealloc_regionEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_8]] : !fir.ref<i32>
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
// CHECK:           %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_10]], %[[VAL_1]] : index
// CHECK:           %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_10]], %[[VAL_1]] : index
// CHECK:           %[[VAL_13:.*]] = fir.alloca !fir.array<?xi32>, %[[VAL_12]] {bindc_name = "a", uniq_name = "_QFlocalizer_with_dealloc_regionEa"}
// CHECK:           %[[VAL_14:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_15:.*]] = fir.declare %[[VAL_13]](%[[VAL_14]]) {uniq_name = "_QFlocalizer_with_dealloc_regionEa"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<?xi32>>
// CHECK:           %[[VAL_16:.*]] = fir.embox %[[VAL_15]](%[[VAL_14]]) : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           fir.store %[[VAL_16]] to %[[VAL_6]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
// CHECK:           fir.do_loop %[[VAL_17:.*]] = %[[VAL_2]] to %[[VAL_10]] step %[[VAL_2]] unordered {

// Local allocation
// CHECK:             %[[VAL_18:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>

// `init` region body
// CHECK:             %[[VAL_19:.*]] = fir.load %[[VAL_6]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
// CHECK:             %[[VAL_20:.*]]:3 = fir.box_dims %[[VAL_19]], %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:             %[[VAL_21:.*]] = fir.shape %[[VAL_20]]#1 : (index) -> !fir.shape<1>
// CHECK:             %[[VAL_22:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_20]]#1 {bindc_name = ".tmp", uniq_name = ""}
// CHECK:             %[[VAL_23:.*]] = fir.declare %[[VAL_22]](%[[VAL_21]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.heap<!fir.array<?xi32>>
// CHECK:             %[[VAL_24:.*]] = fir.embox %[[VAL_23]](%[[VAL_21]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
// CHECK:             %[[VAL_25:.*]] = fir.shape_shift %[[VAL_20]]#0, %[[VAL_20]]#1 : (index, index) -> !fir.shapeshift<1>
// CHECK:             %[[VAL_26:.*]] = fir.rebox %[[VAL_24]](%[[VAL_25]]) : (!fir.box<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?xi32>>
// CHECK:             fir.store %[[VAL_26]] to %[[VAL_18]] : !fir.ref<!fir.box<!fir.array<?xi32>>>

// Loop body
// CHECK:             %[[VAL_27:.*]] = fir.convert %[[VAL_17]] : (index) -> i32
// CHECK:             fir.store %[[VAL_27]] to %[[VAL_5]] : !fir.ref<i32>
// CHECK:             %[[VAL_28:.*]] = fir.declare %[[VAL_18]] {uniq_name = "_QFlocalizer_with_dealloc_regionEa"} : (!fir.ref<!fir.box<!fir.array<?xi32>>>) -> !fir.ref<!fir.box<!fir.array<?xi32>>>
// CHECK:             %[[VAL_29:.*]] = fir.load %[[VAL_28]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
// CHECK:             %[[VAL_30:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (i32) -> i64
// CHECK:             %[[VAL_32:.*]]:3 = fir.box_dims %[[VAL_29]], %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:             %[[VAL_33:.*]] = fir.shift %[[VAL_32]]#0 : (index) -> !fir.shift<1>
// CHECK:             %[[VAL_34:.*]] = fir.array_coor %[[VAL_29]](%[[VAL_33]]) %[[VAL_31]] : (!fir.box<!fir.array<?xi32>>, !fir.shift<1>, i64) -> !fir.ref<i32>
// CHECK:             fir.store %[[VAL_3]] to %[[VAL_34]] : !fir.ref<i32>

// `dealloc` region
// CHECK:             %[[VAL_35:.*]] = fir.load %[[VAL_18]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
// CHECK:             %[[VAL_36:.*]] = fir.box_addr %[[VAL_35]] : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
// CHECK:             %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (!fir.ref<!fir.array<?xi32>>) -> i64
// CHECK:             %[[VAL_38:.*]] = arith.cmpi ne, %[[VAL_37]], %[[VAL_0]] : i64
// CHECK:             fir.if %[[VAL_38]] {
// CHECK:               %[[VAL_39:.*]] = fir.convert %[[VAL_36]] : (!fir.ref<!fir.array<?xi32>>) -> !fir.heap<!fir.array<?xi32>>
// CHECK:               fir.freemem %[[VAL_39]] : !fir.heap<!fir.array<?xi32>>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
