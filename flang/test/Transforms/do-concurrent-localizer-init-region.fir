// Tests converting `fir.local` ops that have `init` regions.

// RUN: fir-opt --split-input-file --simplify-fir-operations %s | FileCheck %s

fir.local {type = local_init} @_QFlocalizer_with_init_regionEp_firstprivate_box_ptr_Uxi32 : !fir.box<!fir.ptr<!fir.array<?xi32>>> init {
^bb0(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, %arg1: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>):
  %c0 = arith.constant 0 : index
  %0 = fir.shape %c0 : (index) -> !fir.shape<1>
  %1 = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
  %2 = fir.embox %1(%0) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
  fir.store %2 to %arg1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  fir.yield(%arg1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
} copy {
^bb0(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, %arg1: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>):
  %0 = fir.load %arg0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  fir.store %0 to %arg1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  fir.yield(%arg1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
}

func.func @_QPlocalizer_with_init_region() {
  %c42_i32 = arith.constant 42 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFlocalizer_with_init_regionEn"}
  %2 = fir.declare %1 {uniq_name = "_QFlocalizer_with_init_regionEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>> {bindc_name = "p", uniq_name = "_QFlocalizer_with_init_regionEp"}
  %4 = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
  %5 = fir.shape %c0 : (index) -> !fir.shape<1>
  %6 = fir.embox %4(%5) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
  fir.store %6 to %3 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %7 = fir.declare %3 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFlocalizer_with_init_regionEp"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %8 = fir.load %2 : !fir.ref<i32>
  %9 = fir.convert %8 : (i32) -> index

  fir.do_concurrent {
    %10 = fir.alloca i32 {bindc_name = "i"}
    %11 = fir.declare %10 {uniq_name = "_QFlocalizer_with_init_regionEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    fir.do_concurrent.loop (%arg0) = (%c1) to (%9) step (%c1) local(@_QFlocalizer_with_init_regionEp_firstprivate_box_ptr_Uxi32 %7 -> %arg1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) {
      %12 = fir.convert %arg0 : (index) -> i32
      fir.store %12 to %11 : !fir.ref<i32>
      %13 = fir.declare %arg1 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFlocalizer_with_init_regionEp"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
      %14 = fir.load %13 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
      %15 = fir.load %11 : !fir.ref<i32>
      %16 = fir.convert %15 : (i32) -> i64
      %17:3 = fir.box_dims %14, %c0 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
      %18 = fir.shift %17#0 : (index) -> !fir.shift<1>
      %19 = fir.array_coor %14(%18) %16 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.shift<1>, i64) -> !fir.ref<i32>
      fir.store %c42_i32 to %19 : !fir.ref<i32>
    }
  }

  return
}

// CHECK-LABEL:   func.func @_QPlocalizer_with_init_region() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 42 : i32
// CHECK:           %[[VAL_3:.*]] = fir.alloca i32 {bindc_name = "i"}
// CHECK:           %[[VAL_4:.*]] = fir.declare %[[VAL_3]] {uniq_name = "_QFlocalizer_with_init_regionEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_5:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_6:.*]] = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFlocalizer_with_init_regionEn"}
// CHECK:           %[[VAL_7:.*]] = fir.declare %[[VAL_6]] {uniq_name = "_QFlocalizer_with_init_regionEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_8:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>> {bindc_name = "p", uniq_name = "_QFlocalizer_with_init_regionEp"}
// CHECK:           %[[VAL_9:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
// CHECK:           %[[VAL_10:.*]] = fir.shape %[[VAL_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_11:.*]] = fir.embox %[[VAL_9]](%[[VAL_10]]) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
// CHECK:           fir.store %[[VAL_11]] to %[[VAL_8]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
// CHECK:           %[[VAL_12:.*]] = fir.declare %[[VAL_8]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFlocalizer_with_init_regionEp"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
// CHECK:           %[[VAL_13:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
// CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i32) -> index
// CHECK:           fir.do_loop %[[VAL_15:.*]] = %[[VAL_1]] to %[[VAL_14]] step %[[VAL_1]] unordered {

// Local allocation
// CHECK:             %[[VAL_16:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>

// `init` region body
// CHECK:             %[[VAL_17:.*]] = fir.shape %[[VAL_0]] : (index) -> !fir.shape<1>
// CHECK:             %[[VAL_18:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
// CHECK:             %[[VAL_19:.*]] = fir.embox %[[VAL_18]](%[[VAL_17]]) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
// CHECK:             fir.store %[[VAL_19]] to %[[VAL_16]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>

// `copy` region body
// CHECK:             %[[VAL_20:.*]] = fir.load %[[VAL_12]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
// CHECK:             fir.store %[[VAL_20]] to %[[VAL_16]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>

// loop body
// CHECK:             %[[VAL_21:.*]] = fir.convert %[[VAL_15]] : (index) -> i32
// CHECK:             fir.store %[[VAL_21]] to %[[VAL_4]] : !fir.ref<i32>
// CHECK:             %[[VAL_22:.*]] = fir.declare %[[VAL_16]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFlocalizer_with_init_regionEp"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
// CHECK:             %[[VAL_23:.*]] = fir.load %[[VAL_22]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
// CHECK:             %[[VAL_24:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
// CHECK:             %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (i32) -> i64
// CHECK:             %[[VAL_26:.*]]:3 = fir.box_dims %[[VAL_23]], %[[VAL_0]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
// CHECK:             %[[VAL_27:.*]] = fir.shift %[[VAL_26]]#0 : (index) -> !fir.shift<1>
// CHECK:             %[[VAL_28:.*]] = fir.array_coor %[[VAL_23]](%[[VAL_27]]) %[[VAL_25]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.shift<1>, i64) -> !fir.ref<i32>
// CHECK:             fir.store %[[VAL_2]] to %[[VAL_28]] : !fir.ref<i32>
// CHECK:           }
// CHECK:           return
// CHECK:         }

