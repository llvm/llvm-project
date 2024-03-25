// Test hlfir.concat operation lowering to operations operating on memory.

// RUN: fir-opt %s -bufferize-hlfir | FileCheck %s

func.func @_QPmax1(%arg0: !fir.boxchar<1> {fir.bindc_name = "c1"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "c2"}, %arg2: !fir.boxchar<1> {fir.bindc_name = "c3"}) {
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1:2 = hlfir.declare %0#0 typeparams %0#1 {uniq_name = "_QFmax1Ec1"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3:2 = hlfir.declare %2#0 typeparams %2#1 {uniq_name = "_QFmax1Ec2"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %4:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %5:2 = hlfir.declare %4#0 typeparams %4#1 {uniq_name = "_QFmax1Ec3"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %6 = hlfir.char_extremum max, %3#0, %5#0 : (!fir.boxchar<1>, !fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.assign %6 to %1#0 : !hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>
  hlfir.destroy %6 : !hlfir.expr<!fir.char<1,?>>
  return
}

// CHECK: func.func @_QPmax1(%[[ARG0:.*]]: !fir.boxchar<1> {fir.bindc_name = "c1"}, %[[ARG1:.*]]: !fir.boxchar<1> {fir.bindc_name = "c2"}, %[[ARG2:.*]]: !fir.boxchar<1> {fir.bindc_name = "c3"}) {
// CHECK:   %[[VAL_0:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:   %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]]#0 typeparams %[[VAL_0]]#1 {uniq_name = "_QFmax1Ec1"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:   %[[VAL_2:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:   %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]]#0 typeparams %[[VAL_2]]#1 {uniq_name = "_QFmax1Ec2"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:   %[[VAL_4:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:   %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]]#0 typeparams %[[VAL_4]]#1 {uniq_name = "_QFmax1Ec3"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:   %[[VAL_6:.*]] = arith.cmpi uge, %[[VAL_2]]#1, %[[VAL_4]]#1 : index
// CHECK:   %[[VAL_7:.*]] = arith.select %[[VAL_6]], %[[VAL_2]]#1, %[[VAL_4]]#1 : index
// CHECK:   %[[VAL_8:.*]] = fir.convert %[[VAL_5]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:   %[[VAL_9:.*]] = fir.convert %[[VAL_3]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:   %[[VAL_10:.*]] = fir.convert %[[VAL_4]]#1 : (index) -> i64
// CHECK:   %[[VAL_11:.*]] = fir.convert %[[VAL_2]]#1 : (index) -> i64
// CHECK:   %[[VAL_12:.*]] = fir.call @_FortranACharacterCompareScalar1(%[[VAL_8]], %[[VAL_9]], %[[VAL_10]], %[[VAL_11]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32
// CHECK:   %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:   %[[VAL_13:.*]] = arith.cmpi sgt, %[[VAL_12]], %[[C0_I32]] : i32
// CHECK:   %[[VAL_14:.*]] = arith.select %[[VAL_13]], %[[VAL_5]]#1, %[[VAL_3]]#1 : !fir.ref<!fir.char<1,?>>
// CHECK:   %[[VAL_15:.*]] = arith.select %[[VAL_13]], %[[VAL_4]]#1, %[[VAL_2]]#1 : index
// CHECK:   %[[VAL_16:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_7]] : index) {bindc_name = ".chrtmp"}
// CHECK:   %[[VAL_17:.*]] = arith.cmpi slt, %[[VAL_7]], %[[VAL_15]] : index
// CHECK:   %[[VAL_18:.*]] = arith.select %[[VAL_17]], %[[VAL_7]], %[[VAL_15]] : index
// CHECK:   %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:   %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (index) -> i64
// CHECK:   %[[VAL_20:.*]] = arith.muli %[[C1_I64]], %[[VAL_19]] : i64
// CHECK:   %[[FALSE:.*]] = arith.constant false
// CHECK:   %[[VAL_21:.*]] = fir.convert %[[VAL_16]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:   %[[VAL_22:.*]] = fir.convert %[[VAL_14]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:   fir.call @llvm.memmove.p0.p0.i64(%[[VAL_21]], %[[VAL_22]], %[[VAL_20]], %[[FALSE]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
// CHECK:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:   %[[VAL_23:.*]] = arith.subi %[[VAL_7]], %[[C1]] : index
// CHECK:   %[[C32_I8:.*]] = arith.constant 32 : i8
// CHECK:   %[[VAL_24:.*]] = fir.undefined !fir.char<1>
// CHECK:   %[[VAL_25:.*]] = fir.insert_value %[[VAL_24]], %[[C32_I8]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
// CHECK:   %[[C1_0:.*]] = arith.constant 1 : index
// CHECK:   fir.do_loop %[[ARG3:.*]] = %[[VAL_18:.*]] to %[[VAL_23]] step %[[C1_0]] {
// CHECK:     %[[VAL_30:.*]] = fir.convert %[[VAL_16]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:     %[[VAL_31:.*]] = fir.coordinate_of %[[VAL_30]], %[[ARG3]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:     fir.store %[[VAL_25:.*]] to %[[VAL_31]] : !fir.ref<!fir.char<1>>
// CHECK:   }
// CHECK:   %[[VAL_26:.*]]:2 = hlfir.declare %[[VAL_16]] typeparams %[[VAL_7]] {uniq_name = ".tmp.char_extremum"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:   %[[FALSE_1:.*]] = arith.constant false
// CHECK:   %[[VAL_27:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:   %[[VAL_28:.*]] = fir.insert_value %[[VAL_27]], %[[FALSE_1]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:   %[[VAL_29:.*]] = fir.insert_value %[[VAL_28]], %[[VAL_26]]#0, [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>
// CHECK:   hlfir.assign %[[VAL_26]]#0 to %[[VAL_1]]#0 : !fir.boxchar<1>, !fir.boxchar<1>
// CHECK:   return
// CHECK: }

func.func @_QPmin1(%arg0: !fir.boxchar<1> {fir.bindc_name = "c1"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "c2"}, %arg2: !fir.boxchar<1> {fir.bindc_name = "c3"}) {
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1:2 = hlfir.declare %0#0 typeparams %0#1 {uniq_name = "_QFmin1Ec1"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3:2 = hlfir.declare %2#0 typeparams %2#1 {uniq_name = "_QFmin1Ec2"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %4:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %5:2 = hlfir.declare %4#0 typeparams %4#1 {uniq_name = "_QFmin1Ec3"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %6 = hlfir.char_extremum min, %3#0, %5#0 : (!fir.boxchar<1>, !fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.assign %6 to %1#0 : !hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>
  hlfir.destroy %6 : !hlfir.expr<!fir.char<1,?>>
  return
}

// CHECK: func.func @_QPmin1(%[[ARG0:.*]]: !fir.boxchar<1> {fir.bindc_name = "c1"}, %[[ARG1:.*]]: !fir.boxchar<1> {fir.bindc_name = "c2"}, %[[ARG2:.*]]: !fir.boxchar<1> {fir.bindc_name = "c3"}) {
// CHECK:   %[[VAL_0:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:   %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]]#0 typeparams %[[VAL_0]]#1 {uniq_name = "_QFmin1Ec1"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:   %[[VAL_2:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:   %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]]#0 typeparams %[[VAL_2]]#1 {uniq_name = "_QFmin1Ec2"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:   %[[VAL_4:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:   %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]]#0 typeparams %[[VAL_4]]#1 {uniq_name = "_QFmin1Ec3"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:   %[[VAL_6:.*]] = arith.cmpi uge, %[[VAL_2]]#1, %[[VAL_4]]#1 : index
// CHECK:   %[[VAL_7:.*]] = arith.select %[[VAL_6]], %[[VAL_2]]#1, %[[VAL_4]]#1 : index
// CHECK:   %[[VAL_8:.*]] = fir.convert %[[VAL_5]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:   %[[VAL_9:.*]] = fir.convert %[[VAL_3]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:   %[[VAL_10:.*]] = fir.convert %[[VAL_4]]#1 : (index) -> i64
// CHECK:   %[[VAL_11:.*]] = fir.convert %[[VAL_2]]#1 : (index) -> i64
// CHECK:   %[[VAL_12:.*]] = fir.call @_FortranACharacterCompareScalar1(%[[VAL_8]], %[[VAL_9]], %[[VAL_10]], %[[VAL_11]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32
// CHECK:   %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:   %[[VAL_13:.*]] = arith.cmpi slt, %[[VAL_12]], %[[C0_I32]] : i32
// CHECK:   %[[VAL_14:.*]] = arith.select %[[VAL_13]], %[[VAL_5]]#1, %[[VAL_3]]#1 : !fir.ref<!fir.char<1,?>>
// CHECK:   %[[VAL_15:.*]] = arith.select %[[VAL_13]], %[[VAL_4]]#1, %[[VAL_2]]#1 : index
// CHECK:   %[[VAL_16:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_7]] : index) {bindc_name = ".chrtmp"}
// CHECK:   %[[VAL_17:.*]] = arith.cmpi slt, %[[VAL_7]], %[[VAL_15]] : index
// CHECK:   %[[VAL_18:.*]] = arith.select %[[VAL_17]], %[[VAL_7]], %[[VAL_15]] : index
// CHECK:   %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:   %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (index) -> i64
// CHECK:   %[[VAL_20:.*]] = arith.muli %[[C1_I64]], %[[VAL_19]] : i64
// CHECK:   %[[FALSE:.*]] = arith.constant false
// CHECK:   %[[VAL_21:.*]] = fir.convert %[[VAL_16]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:   %[[VAL_22:.*]] = fir.convert %[[VAL_14]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:   fir.call @llvm.memmove.p0.p0.i64(%[[VAL_21]], %[[VAL_22]], %[[VAL_20]], %[[FALSE]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
// CHECK:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:   %[[VAL_23:.*]] = arith.subi %[[VAL_7]], %[[C1]] : index
// CHECK:   %[[C32_I8:.*]] = arith.constant 32 : i8
// CHECK:   %[[VAL_24:.*]] = fir.undefined !fir.char<1>
// CHECK:   %[[VAL_25:.*]] = fir.insert_value %[[VAL_24]], %[[C32_I8]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
// CHECK:   %[[C1_0:.*]] = arith.constant 1 : index
// CHECK:   fir.do_loop %[[ARG3:.*]] = %[[VAL_18:.*]] to %[[VAL_23]] step %[[C1_0]] {
// CHECK:     %[[VAL_30:.*]] = fir.convert %[[VAL_16]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:     %[[VAL_31:.*]] = fir.coordinate_of %[[VAL_30]], %[[ARG3]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:     fir.store %[[VAL_25:.*]] to %[[VAL_31]] : !fir.ref<!fir.char<1>>
// CHECK:   }
// CHECK:   %[[VAL_26:.*]]:2 = hlfir.declare %[[VAL_16]] typeparams %[[VAL_7]] {uniq_name = ".tmp.char_extremum"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:   %[[FALSE_1:.*]] = arith.constant false
// CHECK:   %[[VAL_27:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:   %[[VAL_28:.*]] = fir.insert_value %[[VAL_27]], %[[FALSE_1]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:   %[[VAL_29:.*]] = fir.insert_value %[[VAL_28]], %[[VAL_26]]#0, [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>
// CHECK:   hlfir.assign %[[VAL_26]]#0 to %[[VAL_1]]#0 : !fir.boxchar<1>, !fir.boxchar<1>
// CHECK:   return
// CHECK: }

func.func @_QPmax2(%arg0: !fir.boxchar<1> {fir.bindc_name = "c1"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "c2"}, %arg2: !fir.boxchar<1> {fir.bindc_name = "c3"}) {
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1 = fir.convert %0#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,?>>>
  %c100 = arith.constant 100 : index
  %2 = fir.shape %c100 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %1(%2) typeparams %0#1 {uniq_name = "_QFmax2Ec1"} : (!fir.ref<!fir.array<100x!fir.char<1,?>>>, !fir.shape<1>, index) -> (!fir.box<!fir.array<100x!fir.char<1,?>>>, !fir.ref<!fir.array<100x!fir.char<1,?>>>)
  %4:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %c10 = arith.constant 10 : index
  %5 = fir.convert %4#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,10>>>
  %c100_0 = arith.constant 100 : index
  %6 = fir.shape %c100_0 : (index) -> !fir.shape<1>
  %7:2 = hlfir.declare %5(%6) typeparams %c10 {uniq_name = "_QFmax2Ec2"} : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<100x!fir.char<1,10>>>, !fir.ref<!fir.array<100x!fir.char<1,10>>>)
  %8:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %c20 = arith.constant 20 : index
  %9 = fir.convert %8#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,20>>>
  %c100_1 = arith.constant 100 : index
  %10 = fir.shape %c100_1 : (index) -> !fir.shape<1>
  %11:2 = hlfir.declare %9(%10) typeparams %c20 {uniq_name = "_QFmax2Ec3"} : (!fir.ref<!fir.array<100x!fir.char<1,20>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<100x!fir.char<1,20>>>, !fir.ref<!fir.array<100x!fir.char<1,20>>>)
  %c1 = arith.constant 1 : index
  %12 = hlfir.designate %7#0 (%c1)  typeparams %c10 : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, index, index) -> !fir.ref<!fir.char<1,10>>
  %c1_2 = arith.constant 1 : index
  %13 = hlfir.designate %11#0 (%c1_2)  typeparams %c20 : (!fir.ref<!fir.array<100x!fir.char<1,20>>>, index, index) -> !fir.ref<!fir.char<1,20>>
  %14 = hlfir.char_extremum max, %12, %13 : (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,20>>) -> !hlfir.expr<!fir.char<1,20>>
  %c1_3 = arith.constant 1 : index
  %15 = hlfir.designate %3#0 (%c1_3)  typeparams %0#1 : (!fir.box<!fir.array<100x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
  hlfir.assign %14 to %15 : !hlfir.expr<!fir.char<1,20>>, !fir.boxchar<1>
  hlfir.destroy %14 : !hlfir.expr<!fir.char<1,20>>
  return
}

// CHECK: func.func @_QPmax2(%[[ARG0:.*]]: !fir.boxchar<1> {fir.bindc_name = "c1"}, %[[ARG1:.*]]: !fir.boxchar<1> {fir.bindc_name = "c2"}, %[[ARG2:.*]]: !fir.boxchar<1> {fir.bindc_name = "c3"}) {
// CHECK:   %[[VAL_0:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:   %[[VAL_1:.*]] = fir.convert %[[VAL_0]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,?>>>
// CHECK:   %[[C100:.*]] = arith.constant 100 : index
// CHECK:   %[[VAL_2:.*]] = fir.shape %[[C100]] : (index) -> !fir.shape<1>
// CHECK:   %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_2]]) typeparams %[[VAL_0]]#1 {uniq_name = "_QFmax2Ec1"} : (!fir.ref<!fir.array<100x!fir.char<1,?>>>, !fir.shape<1>, index) -> (!fir.box<!fir.array<100x!fir.char<1,?>>>, !fir.ref<!fir.array<100x!fir.char<1,?>>>)
// CHECK:   %[[VAL_4:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:   %[[C10:.*]] = arith.constant 10 : index
// CHECK:   %[[VAL_5:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,10>>>
// CHECK:   %[[C100_0:.*]] = arith.constant 100 : index
// CHECK:   %[[VAL_6:.*]] = fir.shape %[[C100_0]] : (index) -> !fir.shape<1>
// CHECK:   %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_5]](%[[VAL_6]]) typeparams %[[C10]] {uniq_name = "_QFmax2Ec2"} : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<100x!fir.char<1,10>>>, !fir.ref<!fir.array<100x!fir.char<1,10>>>)
// CHECK:   %[[VAL_8:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:   %[[C20:.*]] = arith.constant 20 : index
// CHECK:   %[[VAL_9:.*]] = fir.convert %[[VAL_8]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,20>>>
// CHECK:   %[[C100_1:.*]] = arith.constant 100 : index
// CHECK:   %[[VAL_10:.*]] = fir.shape %[[C100_1]] : (index) -> !fir.shape<1>
// CHECK:   %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_9]](%[[VAL_10]]) typeparams %[[C20]] {uniq_name = "_QFmax2Ec3"} : (!fir.ref<!fir.array<100x!fir.char<1,20>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<100x!fir.char<1,20>>>, !fir.ref<!fir.array<100x!fir.char<1,20>>>)
// CHECK:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:   %[[VAL_12:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[C1]])  typeparams %[[C1]]0 : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, index, index) -> !fir.ref<!fir.char<1,10>>
// CHECK:   %[[C1_2:.*]] = arith.constant 1 : index
// CHECK:   %[[VAL_13:.*]] = hlfir.designate %[[VAL_11]]#0 (%[[C1_2]])  typeparams %[[C20]] : (!fir.ref<!fir.array<100x!fir.char<1,20>>>, index, index) -> !fir.ref<!fir.char<1,20>>
// CHECK:   %[[VAL_14:.*]] = arith.cmpi uge, %[[C10]], %[[C20]] : index
// CHECK:   %[[VAL_15:.*]] = arith.select %[[VAL_14]], %[[C10]], %[[C20]] : index
// CHECK:   %[[VAL_16:.*]] = fir.convert %[[VAL_13]] : (!fir.ref<!fir.char<1,20>>) -> !fir.ref<i8>
// CHECK:   %[[VAL_17:.*]] = fir.convert %[[VAL_12]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
// CHECK:   %[[VAL_18:.*]] = fir.convert %[[C20]] : (index) -> i64
// CHECK:   %[[VAL_19:.*]] = fir.convert %[[C10]] : (index) -> i64
// CHECK:   %[[VAL_20:.*]] = fir.call @_FortranACharacterCompareScalar1(%[[VAL_16]], %[[VAL_17]], %[[VAL_18]], %[[VAL_19]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32
// CHECK:   %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:   %[[VAL_21:.*]] = arith.cmpi sgt, %[[VAL_20]], %[[C0_I32]] : i32
// CHECK:   %[[VAL_22:.*]] = fir.convert %[[VAL_12]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:   %[[VAL_23:.*]] = fir.convert %[[VAL_13]] : (!fir.ref<!fir.char<1,20>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:   %[[VAL_24:.*]] = arith.select %[[VAL_21]], %[[VAL_23]], %[[VAL_22]] : !fir.ref<!fir.char<1,?>>
// CHECK:   %[[VAL_25:.*]] = arith.select %[[VAL_21]], %[[C20]], %[[C10]] : index
// CHECK:   %[[VAL_26:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_15]] : index) {bindc_name = ".chrtmp"}
// CHECK:   %[[VAL_27:.*]] = arith.cmpi slt, %[[VAL_15]], %[[VAL_25]] : index
// CHECK:   %[[VAL_28:.*]] = arith.select %[[VAL_27]], %[[VAL_15]], %[[VAL_25]] : index
// CHECK:   %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:   %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (index) -> i64
// CHECK:   %[[VAL_30:.*]] = arith.muli %[[C1_I64]], %[[VAL_29]] : i64
// CHECK:   %[[FALSE:.*]] = arith.constant false
// CHECK:   %[[VAL_31:.*]] = fir.convert %[[VAL_26]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:   %[[VAL_32:.*]] = fir.convert %[[VAL_24]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:   fir.call @llvm.memmove.p0.p0.i64(%[[VAL_31]], %[[VAL_32]], %[[VAL_30]], %[[FALSE]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
// CHECK:   %[[C1_3:.*]] = arith.constant 1 : index
// CHECK:   %[[VAL_33:.*]] = arith.subi %[[VAL_15]], %[[C1_3]] : index
// CHECK:   %[[C32_I8:.*]] = arith.constant 32 : i8
// CHECK:   %[[VAL_34:.*]] = fir.undefined !fir.char<1>
// CHECK:   %[[VAL_35:.*]] = fir.insert_value %[[VAL_34]], %[[C32_I8]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
// CHECK:   %[[C1_4:.*]] = arith.constant 1 : index
// CHECK:   fir.do_loop %[[ARG3:.*]] = %[[VAL_28:.*]] to %[[VAL_33]] step %[[C1_4]] {
// CHECK:     %[[VAL_42:.*]] = fir.convert %[[VAL_26]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:     %[[VAL_43:.*]] = fir.coordinate_of %[[VAL_42]], %[[ARG3]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:     fir.store %[[VAL_35:.*]] to %[[VAL_43]] : !fir.ref<!fir.char<1>>
// CHECK:   }
// CHECK:   %[[VAL_36:.*]] = fir.convert %[[VAL_26]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,20>>
// CHECK:   %[[VAL_37:.*]]:2 = hlfir.declare %[[VAL_36]] typeparams %[[VAL_15]] {uniq_name = ".tmp.char_extremum"} : (!fir.ref<!fir.char<1,20>>, index) -> (!fir.ref<!fir.char<1,20>>, !fir.ref<!fir.char<1,20>>)
// CHECK:   %[[FALSE_5:.*]] = arith.constant false
// CHECK:   %[[VAL_38:.*]] = fir.undefined tuple<!fir.ref<!fir.char<1,20>>, i1>
// CHECK:   %[[VAL_39:.*]] = fir.insert_value %[[VAL_38]], %[[FALSE_5]], [1 : index] : (tuple<!fir.ref<!fir.char<1,20>>, i1>, i1) -> tuple<!fir.ref<!fir.char<1,20>>, i1>
// CHECK:   %[[VAL_40:.*]] = fir.insert_value %[[VAL_39]], %[[VAL_37]]#0, [0 : index] : (tuple<!fir.ref<!fir.char<1,20>>, i1>, !fir.ref<!fir.char<1,20>>) -> tuple<!fir.ref<!fir.char<1,20>>, i1>
// CHECK:   %[[C1_6:.*]] = arith.constant 1 : index
// CHECK:   %[[VAL_41:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[C1_6]])  typeparams %[[VAL_0]]#1 : (!fir.box<!fir.array<100x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
// CHECK:   hlfir.assign %[[VAL_37]]#0 to %[[VAL_41:.*]] : !fir.ref<!fir.char<1,20>>, !fir.boxchar<1>
// CHECK:   return
// CHECK: }

func.func @_QPmin2(%arg0: !fir.boxchar<1> {fir.bindc_name = "c1"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "c2"}, %arg2: !fir.boxchar<1> {fir.bindc_name = "c3"}) {
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1 = fir.convert %0#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,?>>>
  %c100 = arith.constant 100 : index
  %2 = fir.shape %c100 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %1(%2) typeparams %0#1 {uniq_name = "_QFmin2Ec1"} : (!fir.ref<!fir.array<100x!fir.char<1,?>>>, !fir.shape<1>, index) -> (!fir.box<!fir.array<100x!fir.char<1,?>>>, !fir.ref<!fir.array<100x!fir.char<1,?>>>)
  %4:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %c10 = arith.constant 10 : index
  %5 = fir.convert %4#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,10>>>
  %c100_0 = arith.constant 100 : index
  %6 = fir.shape %c100_0 : (index) -> !fir.shape<1>
  %7:2 = hlfir.declare %5(%6) typeparams %c10 {uniq_name = "_QFmin2Ec2"} : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<100x!fir.char<1,10>>>, !fir.ref<!fir.array<100x!fir.char<1,10>>>)
  %8:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %c20 = arith.constant 20 : index
  %9 = fir.convert %8#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,20>>>
  %c100_1 = arith.constant 100 : index
  %10 = fir.shape %c100_1 : (index) -> !fir.shape<1>
  %11:2 = hlfir.declare %9(%10) typeparams %c20 {uniq_name = "_QFmin2Ec3"} : (!fir.ref<!fir.array<100x!fir.char<1,20>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<100x!fir.char<1,20>>>, !fir.ref<!fir.array<100x!fir.char<1,20>>>)
  %c1 = arith.constant 1 : index
  %12 = hlfir.designate %7#0 (%c1)  typeparams %c10 : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, index, index) -> !fir.ref<!fir.char<1,10>>
  %c1_2 = arith.constant 1 : index
  %13 = hlfir.designate %11#0 (%c1_2)  typeparams %c20 : (!fir.ref<!fir.array<100x!fir.char<1,20>>>, index, index) -> !fir.ref<!fir.char<1,20>>
  %14 = hlfir.char_extremum min, %12, %13 : (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,20>>) -> !hlfir.expr<!fir.char<1,20>>
  %c1_3 = arith.constant 1 : index
  %15 = hlfir.designate %3#0 (%c1_3)  typeparams %0#1 : (!fir.box<!fir.array<100x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
  hlfir.assign %14 to %15 : !hlfir.expr<!fir.char<1,20>>, !fir.boxchar<1>
  hlfir.destroy %14 : !hlfir.expr<!fir.char<1,20>>
  return
}

// CHECK: func.func @_QPmin2(%[[ARG0:.*]]: !fir.boxchar<1> {fir.bindc_name = "c1"}, %[[ARG1:.*]]: !fir.boxchar<1> {fir.bindc_name = "c2"}, %[[ARG2:.*]]: !fir.boxchar<1> {fir.bindc_name = "c3"}) {
// CHECK:   %[[VAL_0:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:   %[[VAL_1:.*]] = fir.convert %[[VAL_0]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,?>>>
// CHECK:   %[[C100:.*]] = arith.constant 100 : index
// CHECK:   %[[VAL_2:.*]] = fir.shape %[[C100]] : (index) -> !fir.shape<1>
// CHECK:   %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_2]]) typeparams %[[VAL_0]]#1 {uniq_name = "_QFmin2Ec1"} : (!fir.ref<!fir.array<100x!fir.char<1,?>>>, !fir.shape<1>, index) -> (!fir.box<!fir.array<100x!fir.char<1,?>>>, !fir.ref<!fir.array<100x!fir.char<1,?>>>)
// CHECK:   %[[VAL_4:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:   %[[C10:.*]] = arith.constant 10 : index
// CHECK:   %[[VAL_5:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,10>>>
// CHECK:   %[[C100_0:.*]] = arith.constant 100 : index
// CHECK:   %[[VAL_6:.*]] = fir.shape %[[C100_0]] : (index) -> !fir.shape<1>
// CHECK:   %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_5]](%[[VAL_6]]) typeparams %[[C10]] {uniq_name = "_QFmin2Ec2"} : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<100x!fir.char<1,10>>>, !fir.ref<!fir.array<100x!fir.char<1,10>>>)
// CHECK:   %[[VAL_8:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:   %[[C20:.*]] = arith.constant 20 : index
// CHECK:   %[[VAL_9:.*]] = fir.convert %[[VAL_8]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,20>>>
// CHECK:   %[[C100_1:.*]] = arith.constant 100 : index
// CHECK:   %[[VAL_10:.*]] = fir.shape %[[C100_1]] : (index) -> !fir.shape<1>
// CHECK:   %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_9]](%[[VAL_10]]) typeparams %[[C20]] {uniq_name = "_QFmin2Ec3"} : (!fir.ref<!fir.array<100x!fir.char<1,20>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<100x!fir.char<1,20>>>, !fir.ref<!fir.array<100x!fir.char<1,20>>>)
// CHECK:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:   %[[VAL_12:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[C1]])  typeparams %[[C1]]0 : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, index, index) -> !fir.ref<!fir.char<1,10>>
// CHECK:   %[[C1_2:.*]] = arith.constant 1 : index
// CHECK:   %[[VAL_13:.*]] = hlfir.designate %[[VAL_11]]#0 (%[[C1_2]])  typeparams %[[C20]] : (!fir.ref<!fir.array<100x!fir.char<1,20>>>, index, index) -> !fir.ref<!fir.char<1,20>>
// CHECK:   %[[VAL_14:.*]] = arith.cmpi uge, %[[C10]], %[[C20]] : index
// CHECK:   %[[VAL_15:.*]] = arith.select %[[VAL_14]], %[[C10]], %[[C20]] : index
// CHECK:   %[[VAL_16:.*]] = fir.convert %[[VAL_13]] : (!fir.ref<!fir.char<1,20>>) -> !fir.ref<i8>
// CHECK:   %[[VAL_17:.*]] = fir.convert %[[VAL_12]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
// CHECK:   %[[VAL_18:.*]] = fir.convert %[[C20]] : (index) -> i64
// CHECK:   %[[VAL_19:.*]] = fir.convert %[[C10]] : (index) -> i64
// CHECK:   %[[VAL_20:.*]] = fir.call @_FortranACharacterCompareScalar1(%[[VAL_16]], %[[VAL_17]], %[[VAL_18]], %[[VAL_19]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32
// CHECK:   %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:   %[[VAL_21:.*]] = arith.cmpi slt, %[[VAL_20]], %[[C0_I32]] : i32
// CHECK:   %[[VAL_22:.*]] = fir.convert %[[VAL_12]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:   %[[VAL_23:.*]] = fir.convert %[[VAL_13]] : (!fir.ref<!fir.char<1,20>>) -> !fir.ref<!fir.char<1,?>>
// CHECK:   %[[VAL_24:.*]] = arith.select %[[VAL_21]], %[[VAL_23]], %[[VAL_22]] : !fir.ref<!fir.char<1,?>>
// CHECK:   %[[VAL_25:.*]] = arith.select %[[VAL_21]], %[[C20]], %[[C10]] : index
// CHECK:   %[[VAL_26:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_15]] : index) {bindc_name = ".chrtmp"}
// CHECK:   %[[VAL_27:.*]] = arith.cmpi slt, %[[VAL_15]], %[[VAL_25]] : index
// CHECK:   %[[VAL_28:.*]] = arith.select %[[VAL_27]], %[[VAL_15]], %[[VAL_25]] : index
// CHECK:   %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:   %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (index) -> i64
// CHECK:   %[[VAL_30:.*]] = arith.muli %[[C1_I64]], %[[VAL_29]] : i64
// CHECK:   %[[FALSE:.*]] = arith.constant false
// CHECK:   %[[VAL_31:.*]] = fir.convert %[[VAL_26]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:   %[[VAL_32:.*]] = fir.convert %[[VAL_24]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:   fir.call @llvm.memmove.p0.p0.i64(%[[VAL_31]], %[[VAL_32]], %[[VAL_30]], %[[FALSE]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
// CHECK:   %[[C1_3:.*]] = arith.constant 1 : index
// CHECK:   %[[VAL_33:.*]] = arith.subi %[[VAL_15]], %[[C1_3]] : index
// CHECK:   %[[C32_I8:.*]] = arith.constant 32 : i8
// CHECK:   %[[VAL_34:.*]] = fir.undefined !fir.char<1>
// CHECK:   %[[VAL_35:.*]] = fir.insert_value %[[VAL_34]], %[[C32_I8]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
// CHECK:   %[[C1_4:.*]] = arith.constant 1 : index
// CHECK:   fir.do_loop %[[ARG3:.*]] = %[[VAL_28:.*]] to %[[VAL_33]] step %[[C1_4]] {
// CHECK:     %[[VAL_42:.*]] = fir.convert %[[VAL_26]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:     %[[VAL_43:.*]] = fir.coordinate_of %[[VAL_42]], %[[ARG3]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:     fir.store %[[VAL_35:.*]] to %[[VAL_43]] : !fir.ref<!fir.char<1>>
// CHECK:   }
// CHECK:   %[[VAL_36:.*]] = fir.convert %[[VAL_26]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,20>>
// CHECK:   %[[VAL_37:.*]]:2 = hlfir.declare %[[VAL_36]] typeparams %[[VAL_15]] {uniq_name = ".tmp.char_extremum"} : (!fir.ref<!fir.char<1,20>>, index) -> (!fir.ref<!fir.char<1,20>>, !fir.ref<!fir.char<1,20>>)
// CHECK:   %[[FALSE_5:.*]] = arith.constant false
// CHECK:   %[[VAL_38:.*]] = fir.undefined tuple<!fir.ref<!fir.char<1,20>>, i1>
// CHECK:   %[[VAL_39:.*]] = fir.insert_value %[[VAL_38]], %[[FALSE_5]], [1 : index] : (tuple<!fir.ref<!fir.char<1,20>>, i1>, i1) -> tuple<!fir.ref<!fir.char<1,20>>, i1>
// CHECK:   %[[VAL_40:.*]] = fir.insert_value %[[VAL_39]], %[[VAL_37]]#0, [0 : index] : (tuple<!fir.ref<!fir.char<1,20>>, i1>, !fir.ref<!fir.char<1,20>>) -> tuple<!fir.ref<!fir.char<1,20>>, i1>
// CHECK:   %[[C1_6:.*]] = arith.constant 1 : index
// CHECK:   %[[VAL_41:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[C1_6]])  typeparams %[[VAL_0]]#1 : (!fir.box<!fir.array<100x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
// CHECK:   hlfir.assign %[[VAL_37]]#0 to %[[VAL_41:.*]] : !fir.ref<!fir.char<1,20>>, !fir.boxchar<1>
// CHECK:   return
// CHECK: }

func.func @_QPmax3(%arg0: !fir.boxchar<1> {fir.bindc_name = "c1"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "c2"}, %arg2: !fir.boxchar<1> {fir.bindc_name = "c3"}, %arg3: !fir.boxchar<1> {fir.bindc_name = "c4"}) {
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1:2 = hlfir.declare %0#0 typeparams %0#1 {uniq_name = "_QFmax3Ec1"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3:2 = hlfir.declare %2#0 typeparams %2#1 {uniq_name = "_QFmax3Ec2"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %4:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %5:2 = hlfir.declare %4#0 typeparams %4#1 {uniq_name = "_QFmax3Ec3"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %6:2 = fir.unboxchar %arg3 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %7:2 = hlfir.declare %6#0 typeparams %6#1 {uniq_name = "_QFmax3Ec4"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %8 = hlfir.char_extremum max, %3#0, %5#0, %7#0 : (!fir.boxchar<1>, !fir.boxchar<1>, !fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.assign %8 to %1#0 : !hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>
  hlfir.destroy %8 : !hlfir.expr<!fir.char<1,?>>
  return
}

// CHECK: func.func @_QPmax3(%[[ARG0:.*]]: !fir.boxchar<1> {fir.bindc_name = "c1"}, %[[ARG1:.*]]: !fir.boxchar<1> {fir.bindc_name = "c2"}, %[[ARG2:.*]]: !fir.boxchar<1> {fir.bindc_name = "c3"}, %[[ARG3:.*]]: !fir.boxchar<1> {fir.bindc_name = "c4"}) {
// CHECK:   %[[VAL_0:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:   %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]]#0 typeparams %[[VAL_0]]#1 {uniq_name = "_QFmax3Ec1"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:   %[[VAL_2:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:   %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]]#0 typeparams %[[VAL_2]]#1 {uniq_name = "_QFmax3Ec2"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:   %[[VAL_4:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:   %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]]#0 typeparams %[[VAL_4]]#1 {uniq_name = "_QFmax3Ec3"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:   %[[VAL_6:.*]]:2 = fir.unboxchar %[[ARG3]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:   %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]]#0 typeparams %[[VAL_6]]#1 {uniq_name = "_QFmax3Ec4"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:   %[[VAL_8:.*]] = arith.cmpi uge, %[[VAL_2]]#1, %[[VAL_4]]#1 : index
// CHECK:   %[[VAL_9:.*]] = arith.select %[[VAL_8]], %[[VAL_2]]#1, %[[VAL_4]]#1 : index
// CHECK:   %[[VAL_10:.*]] = fir.convert %[[VAL_5]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:   %[[VAL_11:.*]] = fir.convert %[[VAL_3]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:   %[[VAL_12:.*]] = fir.convert %[[VAL_4]]#1 : (index) -> i64
// CHECK:   %[[VAL_13:.*]] = fir.convert %[[VAL_2]]#1 : (index) -> i64
// CHECK:   %[[VAL_14:.*]] = fir.call @_FortranACharacterCompareScalar1(%[[VAL_10]], %[[VAL_11]], %[[VAL_12]], %[[VAL_13]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32
// CHECK:   %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:   %[[VAL_15:.*]] = arith.cmpi sgt, %[[VAL_14]], %[[C0_I32]] : i32
// CHECK:   %[[VAL_16:.*]] = arith.select %[[VAL_15]], %[[VAL_5]]#1, %[[VAL_3]]#1 : !fir.ref<!fir.char<1,?>>
// CHECK:   %[[VAL_17:.*]] = arith.select %[[VAL_15]], %[[VAL_4]]#1, %[[VAL_2]]#1 : index
// CHECK:   %[[VAL_18:.*]] = arith.cmpi uge, %[[VAL_9]], %[[VAL_6]]#1 : index
// CHECK:   %[[VAL_19:.*]] = arith.select %[[VAL_18]], %[[VAL_9]], %[[VAL_6]]#1 : index
// CHECK:   %[[VAL_20:.*]] = fir.convert %[[VAL_7]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:   %[[VAL_21:.*]] = fir.convert %[[VAL_16]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:   %[[VAL_22:.*]] = fir.convert %[[VAL_6]]#1 : (index) -> i64
// CHECK:   %[[VAL_23:.*]] = fir.convert %[[VAL_17]] : (index) -> i64
// CHECK:   %[[VAL_24:.*]] = fir.call @_FortranACharacterCompareScalar1(%[[VAL_20]], %[[VAL_21]], %[[VAL_22]], %[[VAL_23]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32
// CHECK:   %[[C0_I32_0:.*]] = arith.constant 0 : i32
// CHECK:   %[[VAL_25:.*]] = arith.cmpi sgt, %[[VAL_24]], %[[C0_I32_0]] : i32
// CHECK:   %[[VAL_26:.*]] = arith.select %[[VAL_25]], %[[VAL_7]]#1, %[[VAL_16]] : !fir.ref<!fir.char<1,?>>
// CHECK:   %[[VAL_27:.*]] = arith.select %[[VAL_25]], %[[VAL_6]]#1, %[[VAL_17]] : index
// CHECK:   %[[VAL_28:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_19]] : index) {bindc_name = ".chrtmp"}
// CHECK:   %[[VAL_29:.*]] = arith.cmpi slt, %[[VAL_19]], %[[VAL_27]] : index
// CHECK:   %[[VAL_30:.*]] = arith.select %[[VAL_29]], %[[VAL_19]], %[[VAL_27]] : index
// CHECK:   %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:   %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (index) -> i64
// CHECK:   %[[VAL_32:.*]] = arith.muli %[[C1_I64]], %[[VAL_31]] : i64
// CHECK:   %[[FALSE:.*]] = arith.constant false
// CHECK:   %[[VAL_33:.*]] = fir.convert %[[VAL_28]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:   %[[VAL_34:.*]] = fir.convert %[[VAL_26]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:   fir.call @llvm.memmove.p0.p0.i64(%[[VAL_33]], %[[VAL_34]], %[[VAL_32]], %[[FALSE]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
// CHECK:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:   %[[VAL_35:.*]] = arith.subi %[[VAL_19]], %[[C1]] : index
// CHECK:   %[[C32_I8:.*]] = arith.constant 32 : i8
// CHECK:   %[[VAL_36:.*]] = fir.undefined !fir.char<1>
// CHECK:   %[[VAL_37:.*]] = fir.insert_value %[[VAL_36]], %[[C32_I8]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
// CHECK:   %[[C1_1:.*]] = arith.constant 1 : index
// CHECK:   fir.do_loop %[[ARG4:.*]] = %[[VAL_30:.*]] to %[[VAL_35]] step %[[C1_1]] {
// CHECK:     %[[VAL_42:.*]] = fir.convert %[[VAL_28]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:     %[[VAL_43:.*]] = fir.coordinate_of %[[VAL_42]], %[[ARG4]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:     fir.store %[[VAL_37:.*]] to %[[VAL_43]] : !fir.ref<!fir.char<1>>
// CHECK:   }
// CHECK:   %[[VAL_38:.*]]:2 = hlfir.declare %[[VAL_28]] typeparams %[[VAL_19]] {uniq_name = ".tmp.char_extremum"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:   %[[FALSE_2:.*]] = arith.constant false
// CHECK:   %[[VAL_39:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:   %[[VAL_40:.*]] = fir.insert_value %[[VAL_39]], %[[FALSE_2]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:   %[[VAL_41:.*]] = fir.insert_value %[[VAL_40]], %[[VAL_38]]#0, [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>
// CHECK:   hlfir.assign %[[VAL_38]]#0 to %[[VAL_1]]#0 : !fir.boxchar<1>, !fir.boxchar<1>
// CHECK:   return
// CHECK: }

func.func @_QPmin3(%arg0: !fir.boxchar<1> {fir.bindc_name = "c1"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "c2"}, %arg2: !fir.boxchar<1> {fir.bindc_name = "c3"}, %arg3: !fir.boxchar<1> {fir.bindc_name = "c4"}) {
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1:2 = hlfir.declare %0#0 typeparams %0#1 {uniq_name = "_QFmin3Ec1"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3:2 = hlfir.declare %2#0 typeparams %2#1 {uniq_name = "_QFmin3Ec2"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %4:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %5:2 = hlfir.declare %4#0 typeparams %4#1 {uniq_name = "_QFmin3Ec3"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %6:2 = fir.unboxchar %arg3 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %7:2 = hlfir.declare %6#0 typeparams %6#1 {uniq_name = "_QFmin3Ec4"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %8 = hlfir.char_extremum min, %3#0, %5#0, %7#0 : (!fir.boxchar<1>, !fir.boxchar<1>, !fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.assign %8 to %1#0 : !hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>
  hlfir.destroy %8 : !hlfir.expr<!fir.char<1,?>>
  return
}

// CHECK: func.func @_QPmin3(%[[ARG0:.*]]: !fir.boxchar<1> {fir.bindc_name = "c1"}, %[[ARG1:.*]]: !fir.boxchar<1> {fir.bindc_name = "c2"}, %[[ARG2:.*]]: !fir.boxchar<1> {fir.bindc_name = "c3"}, %[[ARG3:.*]]: !fir.boxchar<1> {fir.bindc_name = "c4"}) {
// CHECK:   %[[VAL_0:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:   %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]]#0 typeparams %[[VAL_0]]#1 {uniq_name = "_QFmin3Ec1"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:   %[[VAL_2:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:   %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]]#0 typeparams %[[VAL_2]]#1 {uniq_name = "_QFmin3Ec2"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:   %[[VAL_4:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:   %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]]#0 typeparams %[[VAL_4]]#1 {uniq_name = "_QFmin3Ec3"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:   %[[VAL_6:.*]]:2 = fir.unboxchar %[[ARG3]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:   %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]]#0 typeparams %[[VAL_6]]#1 {uniq_name = "_QFmin3Ec4"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:   %[[VAL_8:.*]] = arith.cmpi uge, %[[VAL_2]]#1, %[[VAL_4]]#1 : index
// CHECK:   %[[VAL_9:.*]] = arith.select %[[VAL_8]], %[[VAL_2]]#1, %[[VAL_4]]#1 : index
// CHECK:   %[[VAL_10:.*]] = fir.convert %[[VAL_5]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:   %[[VAL_11:.*]] = fir.convert %[[VAL_3]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:   %[[VAL_12:.*]] = fir.convert %[[VAL_4]]#1 : (index) -> i64
// CHECK:   %[[VAL_13:.*]] = fir.convert %[[VAL_2]]#1 : (index) -> i64
// CHECK:   %[[VAL_14:.*]] = fir.call @_FortranACharacterCompareScalar1(%[[VAL_10]], %[[VAL_11]], %[[VAL_12]], %[[VAL_13]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32
// CHECK:   %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:   %[[VAL_15:.*]] = arith.cmpi slt, %[[VAL_14]], %[[C0_I32]] : i32
// CHECK:   %[[VAL_16:.*]] = arith.select %[[VAL_15]], %[[VAL_5]]#1, %[[VAL_3]]#1 : !fir.ref<!fir.char<1,?>>
// CHECK:   %[[VAL_17:.*]] = arith.select %[[VAL_15]], %[[VAL_4]]#1, %[[VAL_2]]#1 : index
// CHECK:   %[[VAL_18:.*]] = arith.cmpi uge, %[[VAL_9]], %[[VAL_6]]#1 : index
// CHECK:   %[[VAL_19:.*]] = arith.select %[[VAL_18]], %[[VAL_9]], %[[VAL_6]]#1 : index
// CHECK:   %[[VAL_20:.*]] = fir.convert %[[VAL_7]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:   %[[VAL_21:.*]] = fir.convert %[[VAL_16]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:   %[[VAL_22:.*]] = fir.convert %[[VAL_6]]#1 : (index) -> i64
// CHECK:   %[[VAL_23:.*]] = fir.convert %[[VAL_17]] : (index) -> i64
// CHECK:   %[[VAL_24:.*]] = fir.call @_FortranACharacterCompareScalar1(%[[VAL_20]], %[[VAL_21]], %[[VAL_22]], %[[VAL_23]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32
// CHECK:   %[[C0_I32_0:.*]] = arith.constant 0 : i32
// CHECK:   %[[VAL_25:.*]] = arith.cmpi slt, %[[VAL_24]], %[[C0_I32_0]] : i32
// CHECK:   %[[VAL_26:.*]] = arith.select %[[VAL_25]], %[[VAL_7]]#1, %[[VAL_16]] : !fir.ref<!fir.char<1,?>>
// CHECK:   %[[VAL_27:.*]] = arith.select %[[VAL_25]], %[[VAL_6]]#1, %[[VAL_17]] : index
// CHECK:   %[[VAL_28:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_19]] : index) {bindc_name = ".chrtmp"}
// CHECK:   %[[VAL_29:.*]] = arith.cmpi slt, %[[VAL_19]], %[[VAL_27]] : index
// CHECK:   %[[VAL_30:.*]] = arith.select %[[VAL_29]], %[[VAL_19]], %[[VAL_27]] : index
// CHECK:   %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:   %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (index) -> i64
// CHECK:   %[[VAL_32:.*]] = arith.muli %[[C1_I64]], %[[VAL_31]] : i64
// CHECK:   %[[FALSE:.*]] = arith.constant false
// CHECK:   %[[VAL_33:.*]] = fir.convert %[[VAL_28]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:   %[[VAL_34:.*]] = fir.convert %[[VAL_26]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:   fir.call @llvm.memmove.p0.p0.i64(%[[VAL_33]], %[[VAL_34]], %[[VAL_32]], %[[FALSE]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
// CHECK:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:   %[[VAL_35:.*]] = arith.subi %[[VAL_19]], %[[C1]] : index
// CHECK:   %[[C32_I8:.*]] = arith.constant 32 : i8
// CHECK:   %[[VAL_36:.*]] = fir.undefined !fir.char<1>
// CHECK:   %[[VAL_37:.*]] = fir.insert_value %[[VAL_36]], %[[C32_I8]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
// CHECK:   %[[C1_1:.*]] = arith.constant 1 : index
// CHECK:   fir.do_loop %[[ARG4:.*]] = %[[VAL_30:.*]] to %[[VAL_35]] step %[[C1_1]] {
// CHECK:     %[[VAL_42:.*]] = fir.convert %[[VAL_28]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:     %[[VAL_43:.*]] = fir.coordinate_of %[[VAL_42]], %[[ARG4]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:     fir.store %[[VAL_37:.*]] to %[[VAL_43]] : !fir.ref<!fir.char<1>>
// CHECK:   }
// CHECK:   %[[VAL_38:.*]]:2 = hlfir.declare %[[VAL_28]] typeparams %[[VAL_19]] {uniq_name = ".tmp.char_extremum"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:   %[[FALSE_2:.*]] = arith.constant false
// CHECK:   %[[VAL_39:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:   %[[VAL_40:.*]] = fir.insert_value %[[VAL_39]], %[[FALSE_2]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:   %[[VAL_41:.*]] = fir.insert_value %[[VAL_40]], %[[VAL_38]]#0, [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>
// CHECK:   hlfir.assign %[[VAL_38]]#0 to %[[VAL_1]]#0 : !fir.boxchar<1>, !fir.boxchar<1>
// CHECK:   return
// CHECK: }
