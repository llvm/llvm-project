// RUN: fir-opt --canonicalize %s -split-input-file | FileCheck %s

// CHECK-LABEL: func @check_no_folding
func.func @check_no_folding(%arg0 : !fir.ref<!fir.array<?xi32>>) {
  %0 = fir.alloca i32 {adapt.valuebyref}
  %c1_i32 = arith.constant 1 : i32
  %1 = fir.load %0 : !fir.ref<i32>
  %2 = arith.addi %1, %c1_i32 : i32
  %3 = fir.convert %2 : (i32) -> index
  %c1 = arith.constant 1 : index
  %5 = arith.subi %3, %c1 : index
  %6 = fir.shape %3 : (index) -> !fir.shape<1>
  %7 = fir.slice %3, %5, %3 : (index, index, index) -> !fir.slice<1>
  %8 = fir.embox %arg0(%6) [%7] : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xi32>>
  %9 = fir.box_addr %8 : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
  // CHECK: %[[BOX_ADDR:.*]] = fir.box_addr
  // CHECK: fir.call @check(%[[BOX_ADDR]])
  fir.call @check(%9) : (!fir.ref<!fir.array<?xi32>>) -> ()
  return
}

func.func @check(%arg0: !fir.ref<!fir.array<?xi32>>) {
  return
}

// -----

// CHECK-LABEL: func @check_folding
func.func @check_folding(%arg0 : !fir.ref<!fir.array<?xi32>>) {
  %0 = fir.alloca i32 {adapt.valuebyref}
  %c1_i32 = arith.constant 1 : i32
  %1 = fir.load %0 : !fir.ref<i32>
  %2 = arith.addi %1, %c1_i32 : i32
  %3 = fir.convert %2 : (i32) -> index
  %c1 = arith.constant 1 : index
  %6 = fir.shape %3 : (index) -> !fir.shape<1>
  %7 = fir.embox %arg0(%6) : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
  %8 = fir.box_addr %7 : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
  // CHECK-NOT: fir.box_addr
  fir.call @check(%8) : (!fir.ref<!fir.array<?xi32>>) -> ()
  return
}

func.func @check(%arg0: !fir.ref<!fir.array<?xi32>>) {
  return
}
