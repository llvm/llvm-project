// RUN: fir-opt --stack-arrays %s | FileCheck %s

// Simplest transformation
func.func @simple() {
  %0 = fir.allocmem !fir.array<42xi32>
  fir.freemem %0 : !fir.heap<!fir.array<42xi32>>
  return
}
// CHECK: func.func @simple() {
// CHECK-NEXT: fir.alloca !fir.array<42xi32>
// CHECK-NEXT: return
// CHECK-NEXT: }

// Check fir.must_be_heap allocations are not moved
func.func @must_be_heap() {
  %0 = fir.allocmem !fir.array<42xi32> {fir.must_be_heap = true}
  fir.freemem %0 : !fir.heap<!fir.array<42xi32>>
  return
}
// CHECK:      func.func @must_be_heap() {
// CHECK-NEXT:   %[[ALLOC:.*]] = fir.allocmem !fir.array<42xi32> {fir.must_be_heap = true}
// CHECK-NEXT:   fir.freemem %[[ALLOC]] : !fir.heap<!fir.array<42xi32>>
// CHECK-NEXT:   return
// CHECK-NEXT: }

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
// CHECK:      func.func @dfa1(%arg0: !fir.ref<!fir.logical<4>> {fir.bindc_name = "cond"}) {
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

// Check scf.if (fir.if is not considered a branch operation)
func.func @dfa2(%arg0: i1) {
  %a = fir.allocmem !fir.array<1xi8>
  scf.if %arg0 {
    fir.freemem %a : !fir.heap<!fir.array<1xi8>>
  } else {
  }
  return
}
// CHECK:     func.func @dfa2(%arg0: i1) {
// CHECK-NEXT:  %[[MEM:.*]] = fir.allocmem !fir.array<1xi8>
// CHECK-NEXT:  scf.if %arg0 {
// CHECK-NEXT:    fir.freemem %[[MEM]] : !fir.heap<!fir.array<1xi8>>
// CHECK-NEXT:  } else {
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:  }

// check the alloca is placed after all operands become available
func.func @placement1() {
  // do some stuff with other ssa values
  %1 = arith.constant 1 : index
  %2 = arith.constant 2 : index
  %3 = arith.addi %1, %2 : index
  // operand is now available
  %4 = fir.allocmem !fir.array<?xi32>, %3
  // ...
  fir.freemem %4 : !fir.heap<!fir.array<?xi32>>
  return
}
// CHECK:      func.func @placement1() {
// CHECK-NEXT:   %[[ONE:.*]] = arith.constant 1 : index
// CHECK-NEXT:   %[[TWO:.*]] = arith.constant 2 : index
// CHECK-NEXT:   %[[ARG:.*]] = arith.addi %[[ONE]], %[[TWO]] : index
// CHECK-NEXT:   %[[MEM:.*]] = fir.alloca !fir.array<?xi32>, %[[ARG]]
// CHECK-NEXT:   return
// CHECK-NEXT: }

// check that if there are no operands, then the alloca is placed early
func.func @placement2() {
  // do some stuff with other ssa values
  %1 = arith.constant 1 : index
  %2 = arith.constant 2 : index
  %3 = arith.addi %1, %2 : index
  %4 = fir.allocmem !fir.array<42xi32>
  // ...
  fir.freemem %4 : !fir.heap<!fir.array<42xi32>>
  return
}
// CHECK:      func.func @placement2() {
// CHECK-NEXT:   %[[MEM:.*]] = fir.alloca !fir.array<42xi32>
// CHECK-NEXT:   %[[ONE:.*]] = arith.constant 1 : index
// CHECK-NEXT:   %[[TWO:.*]] = arith.constant 2 : index
// CHECK-NEXT:   %[[SUM:.*]] = arith.addi %[[ONE]], %[[TWO]] : index
// CHECK-NEXT:   return
// CHECK-NEXT: }

// check that stack allocations which must be placed in loops use stacksave
func.func @placement3() {
  %c1 = arith.constant 1 : index
  %c1_i32 = fir.convert %c1 : (index) -> i32
  %c2 = arith.constant 2 : index
  %c10 = arith.constant 10 : index
  %0:2 = fir.do_loop %arg0 = %c1 to %c10 step %c1 iter_args(%arg1 = %c1_i32) -> (index, i32) {
    %3 = arith.addi %c1, %c2 : index
    // operand is now available
    %4 = fir.allocmem !fir.array<?xi32>, %3
    // ...
    fir.freemem %4 : !fir.heap<!fir.array<?xi32>>
    fir.result %3, %c1_i32 : index, i32
  }
  return
}
// CHECK:      func.func @placement3() {
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   %[[C1_I32:.*]] = fir.convert %[[C1]] : (index) -> i32
// CHECK-NEXT:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-NEXT:   %[[C10:.*]] = arith.constant 10 : index
// CHECK-NEXT:   fir.do_loop
// CHECK-NEXT:     %[[SUM:.*]] = arith.addi %[[C1]], %[[C2]] : index
// CHECK-NEXT:     %[[SP:.*]] = fir.call @llvm.stacksave() : () -> !fir.ref<i8>
// CHECK-NEXT:     %[[MEM:.*]] = fir.alloca !fir.array<?xi32>, %[[SUM]]
// CHECK-NEXT:     fir.call @llvm.stackrestore(%[[SP]])
// CHECK-NEXT:     fir.result
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// check that stack save/restore are used in CFG loops
func.func @placement4(%arg0 : i1) {
  %c1 = arith.constant 1 : index
  %c1_i32 = fir.convert %c1 : (index) -> i32
  %c2 = arith.constant 2 : index
  %c10 = arith.constant 10 : index
  cf.br ^bb1
^bb1:
  %3 = arith.addi %c1, %c2 : index
  // operand is now available
  %4 = fir.allocmem !fir.array<?xi32>, %3
  // ...
  fir.freemem %4 : !fir.heap<!fir.array<?xi32>>
  cf.cond_br %arg0, ^bb1, ^bb2
^bb2:
  return
}
// CHECK:      func.func @placement4(%arg0: i1) {
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   %[[C1_I32:.*]] = fir.convert %[[C1]] : (index) -> i32
// CHECK-NEXT:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-NEXT:   %[[C10:.*]] = arith.constant 10 : index
// CHECK-NEXT:   cf.br ^bb1
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   %[[SUM:.*]] = arith.addi %[[C1]], %[[C2]] : index
// CHECK-NEXT:   %[[SP:.*]] = fir.call @llvm.stacksave() : () -> !fir.ref<i8>
// CHECK-NEXT:   %[[MEM:.*]] = fir.alloca !fir.array<?xi32>, %[[SUM]]
// CHECK-NEXT:   fir.call @llvm.stackrestore(%[[SP]]) : (!fir.ref<i8>) -> ()
// CHECK-NEXT:   cf.cond_br %arg0, ^bb1, ^bb2
// CHECK-NEXT: ^bb2:
// CHECK-NEXT:   return
// CHECK-NEXT: }

// check that stacksave is not used when there is an intervening alloca
func.func @placement5() {
  %c1 = arith.constant 1 : index
  %c1_i32 = fir.convert %c1 : (index) -> i32
  %c2 = arith.constant 2 : index
  %c10 = arith.constant 10 : index
  %0:2 = fir.do_loop %arg0 = %c1 to %c10 step %c1 iter_args(%arg1 = %c1_i32) -> (index, i32) {
    %3 = arith.addi %c1, %c2 : index
    // operand is now available
    %4 = fir.allocmem !fir.array<?xi32>, %3
    %5 = fir.alloca i32
    fir.freemem %4 : !fir.heap<!fir.array<?xi32>>
    fir.result %3, %c1_i32 : index, i32
  }
  return
}
// CHECK:      func.func @placement5() {
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   %[[C1_I32:.*]] = fir.convert %[[C1]] : (index) -> i32
// CHECK-NEXT:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-NEXT:   %[[C10:.*]] = arith.constant 10 : index
// CHECK-NEXT:   fir.do_loop
// CHECK-NEXT:     %[[SUM:.*]] = arith.addi %[[C1]], %[[C2]] : index
// CHECK-NEXT:     %[[MEM:.*]] = fir.allocmem !fir.array<?xi32>, %[[SUM]]
// CHECK-NEXT:     %[[IDX:.*]] = fir.alloca i32
// CHECK-NEXT:     fir.freemem %[[MEM]] : !fir.heap<!fir.array<?xi32>>
// CHECK-NEXT:     fir.result
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// check that stack save/restore are not used when the memalloc and freemem are
// in different blocks
func.func @placement6(%arg0: i1) {
  %c1 = arith.constant 1 : index
  %c1_i32 = fir.convert %c1 : (index) -> i32
  %c2 = arith.constant 2 : index
  %c10 = arith.constant 10 : index
  cf.br ^bb1
^bb1:
  %3 = arith.addi %c1, %c2 : index
  // operand is now available
  %4 = fir.allocmem !fir.array<?xi32>, %3
  // ...
  cf.cond_br %arg0, ^bb2, ^bb3
^bb2:
  // ...
  fir.freemem %4 : !fir.heap<!fir.array<?xi32>>
  cf.br ^bb1
^bb3:
  // ...
  fir.freemem %4 : !fir.heap<!fir.array<?xi32>>
  cf.br ^bb1
}
// CHECK:      func.func @placement6(%arg0: i1) {
// CHECK-NEXT:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   %[[c1_i32:.*]] = fir.convert %[[c1]] : (index) -> i32
// CHECK-NEXT:   %[[c2:.*]] = arith.constant 2 : index
// CHECK-NEXT:   %[[c10:.*]] = arith.constant 10 : index
// CHECK-NEXT:   cf.br ^bb1
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   %[[ADD:.*]] = arith.addi %[[c1]], %[[c2]] : index
// CHECK-NEXT:   %[[MEM:.*]] = fir.allocmem !fir.array<?xi32>, %[[ADD]]
// CHECK-NEXT:   cf.cond_br %arg0, ^bb2, ^bb3
// CHECK-NEXT: ^bb2:
// CHECK-NEXT:   fir.freemem %[[MEM]] : !fir.heap<!fir.array<?xi32>>
// CHECK-NEXT:   cf.br ^bb1
// CHECK-NEXT: ^bb3:
// CHECK-NEXT:   fir.freemem %[[MEM]] : !fir.heap<!fir.array<?xi32>>
// CHECK-NEXT:   cf.br ^bb1
// CHECK-NEXT: }

// Check multiple returns, where the memory is always freed
func.func @returns(%arg0: i1) {
  %0 = fir.allocmem !fir.array<42xi32>
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  fir.freemem %0 : !fir.heap<!fir.array<42xi32>>
  return
^bb2:
  fir.freemem %0 : !fir.heap<!fir.array<42xi32>>
  return
}
// CHECK:      func.func @returns(%[[COND:.*]]: i1) {
// CHECK-NEXT:   %[[ALLOC:.*]] = fir.alloca !fir.array<42xi32>
// CHECK-NEXT:   cf.cond_br %[[COND]], ^bb1, ^bb2
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   return
// CHECK-NEXT: ^bb2:
// CHECK-NEXT:   return
// CHECK-NEXT: }

// Check multiple returns, where the memory is not freed on one branch
func.func @returns2(%arg0: i1) {
  %0 = fir.allocmem !fir.array<42xi32>
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  fir.freemem %0 : !fir.heap<!fir.array<42xi32>>
  return
^bb2:
  return
}
// CHECK:      func.func @returns2(%[[COND:.*]]: i1) {
// CHECK-NEXT:   %[[ALLOC:.*]] = fir.allocmem !fir.array<42xi32>
// CHECK-NEXT:   cf.cond_br %[[COND]], ^bb1, ^bb2
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   fir.freemem %[[ALLOC]] : !fir.heap<!fir.array<42xi32>>
// CHECK-NEXT:   return
// CHECK-NEXT: ^bb2:
// CHECK-NEXT:   return
// CHECK-NEXT: }

// Check allocations are not moved outside of an omp region
func.func @omp_placement1() {
  omp.sections {
    omp.section {
      %mem = fir.allocmem !fir.array<42xi32>
      fir.freemem %mem : !fir.heap<!fir.array<42xi32>>
      omp.terminator
    }
    omp.terminator
  }
  return
}
// CHECK:      func.func @omp_placement1() {
// CHECK-NEXT:   omp.sections {
// CHECK-NEXT:     omp.section {
// CHECK-NEXT:       %[[MEM:.*]] = fir.allocmem !fir.array<42xi32>
// TODO: this allocation should be moved to the stack. Unfortunately, the data
// flow analysis fails to propogate the lattice out of the omp region to the
// return satement.
// CHECK-NEXT:       fir.freemem %[[MEM]] : !fir.heap<!fir.array<42xi32>>
// CHECK-NEXT:       omp.terminator
// CHECK-NEXT:     }
// CHECK-NEXT:     omp.terminator
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
