// RUN: fir-opt %s --opt-bufferization | FileCheck %s 

// Check that OpenACC attributes are propagated to the defining operations when
// fir.box_addr is folded in bufferization optimization.

func.func @_QPsub1(%arg0: !fir.ref<!fir.array<?x?xf32>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n1"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n2"}) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.declare %arg1 {uniq_name = "_QFsub1En1"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg2 {uniq_name = "_QFsub1En2"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.load %0 : !fir.ref<i32>
  %3 = fir.convert %2 : (i32) -> index
  %4 = arith.cmpi sgt, %3, %c0 : index
  %5 = arith.select %4, %3, %c0 : index
  %6 = fir.load %1 : !fir.ref<i32>
  %7 = fir.convert %6 : (i32) -> index
  %8 = arith.cmpi sgt, %7, %c0 : index
  %9 = arith.select %8, %7, %c0 : index
  %10 = fir.shape %5, %9 : (index, index) -> !fir.shape<2>
  %11 = fir.declare %arg0(%10) {uniq_name = "_QFsub1Ea"} : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.ref<!fir.array<?x?xf32>>
  %12 = fir.embox %11(%10) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf32>>
  %13:3 = fir.box_dims %12, %c0 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
  %14 = arith.subi %13#1, %c1 : index
  %15 = acc.bounds lowerbound(%c0 : index) upperbound(%14 : index) extent(%13#1 : index) stride(%13#2 : index) startIdx(%c1 : index) {strideInBytes = true}
  %16 = arith.muli %13#2, %13#1 : index
  %17:3 = fir.box_dims %12, %c1 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
  %18 = arith.subi %17#1, %c1 : index
  %19 = acc.bounds lowerbound(%c0 : index) upperbound(%18 : index) extent(%17#1 : index) stride(%16 : index) startIdx(%c1 : index) {strideInBytes = true}
  %20 = fir.box_addr %12 {acc.declare = #acc.declare<dataClause =  acc_present>} : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
  %21 = acc.present varPtr(%20 : !fir.ref<!fir.array<?x?xf32>>) bounds(%15, %19) -> !fir.ref<!fir.array<?x?xf32>> {name = "a"}
  %22 = acc.declare_enter dataOperands(%21 : !fir.ref<!fir.array<?x?xf32>>)
  acc.declare_exit token(%22)
  return
}

// CHECK-LABEL: func.func @_QPsub1(
// CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.array<?x?xf32>> {fir.bindc_name = "a"}
// CHECK: %[[DECL:.*]] = fir.declare %[[ARG0]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_present>, uniq_name = "_QFsub1Ea"} : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.ref<!fir.array<?x?xf32>>
// CHECK: %[[PRES:.*]] = acc.present varPtr(%[[DECL]] : !fir.ref<!fir.array<?x?xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<?x?xf32>> {name = "a"}
// CHECK: %{{.*}} = acc.declare_enter dataOperands(%[[PRES]] : !fir.ref<!fir.array<?x?xf32>>)
