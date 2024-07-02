// RUN: fir-opt --stack-arrays --debug-only=stack-arrays %s 2>&1 | grep -v '\-\-' | FileCheck %s

// Check the data-flow-analysis can detect cases where we aren't sure if memory
// is freed by the end of the function
func.func @dfa1(%arg0: !fir.ref<!fir.logical<4>> {fir.bindc_name = "cond"}) {
 %7 = arith.constant 42 : index
 %8 = fir.allocmem !fir.array<?xi32>, %7 {uniq_name = "_QFdfa1Earr.alloc"}
 %9 = fir.load %arg0 : !fir.ref<!fir.logical<4>>
 %10 = fir.convert %9 : (!fir.logical<4>) -> i1
 fir.if %10 {
   fir.freemem %8 : !fir.heap<!fir.array<?xi32>>
 } else {
 }
 return
}

// 8 visits:
// CHECK: StackArrays: Visiting operation:
// CHECK-NEXT: StackArrays: Visiting operation:
// CHECK-NEXT: StackArrays: Visiting operation:
// CHECK-NEXT: StackArrays: Visiting operation:
// CHECK-NEXT: StackArrays: Visiting operation:
// CHECK-NEXT: StackArrays: Visiting operation:
// CHECK-NEXT: StackArrays: Visiting operation:
// CHECK-NEXT: StackArrays: Visiting operation:
///CHECK-NEXT: module {
// CHECK-NEXT:   func.func @dfa1(%arg0: !fir.ref<!fir.logical<4>> {fir.bindc_name = "cond"}) {
// CHECK-NEXT:   %[[C42:.*]] = arith.constant 42 : index
// CHECK-NEXT:   %[[MEM:.*]] = fir.allocmem !fir.array<?xi32>, %[[C42]] {uniq_name = "_QFdfa1Earr.alloc"}
// CHECK-NEXT:   %[[LOGICAL:.*]] = fir.load %arg0 : !fir.ref<!fir.logical<4>>
// CHECK-NEXT:   %[[BOOL:.*]] = fir.convert %[[LOGICAL]] : (!fir.logical<4>) -> i1
// CHECK-NEXT:   fir.if %[[BOOL]] {
// CHECK-NEXT:     fir.freemem %[[MEM]] : !fir.heap<!fir.array<?xi32>>
// CHECK-NEXT:   } else {
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
// CHECK-NEXT: }

