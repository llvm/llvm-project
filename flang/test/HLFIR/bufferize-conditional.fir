// Test hlfir.conditional bufferization (ConditionalOpConversion).
// RUN: fir-opt --bufferize-hlfir %s | FileCheck %s

// --- Test 1: Scalar non-polymorphic derived type ---
// Both branches yield a !fir.ref<!fir.type<T>> variable. A non-move as_expr
// copies each into a stack temp (mustFree=false); no cast needed.
func.func @test_scalar_derived(%cond: i1, %a: !fir.ref<!fir.type<_Tt{x:i32}>>, %b: !fir.ref<!fir.type<_Tt{x:i32}>>) {
  %0 = hlfir.conditional %cond : (i1) -> !hlfir.expr<!fir.type<_Tt{x:i32}>> {
    hlfir.yield %a : !fir.ref<!fir.type<_Tt{x:i32}>>
  } else {
    hlfir.yield %b : !fir.ref<!fir.type<_Tt{x:i32}>>
  }
  hlfir.destroy %0 : !hlfir.expr<!fir.type<_Tt{x:i32}>>
  return
}
// CHECK-LABEL: func.func @test_scalar_derived(
// CHECK-SAME:    %[[COND:.*]]: i1, %[[A:.*]]: !fir.ref<!fir.type<_Tt{x:i32}>>, %[[B:.*]]: !fir.ref<!fir.type<_Tt{x:i32}>>)
// CHECK:       %[[IF:.*]]:2 = fir.if %[[COND]] -> (!fir.ref<!fir.type<_Tt{x:i32}>>, i1) {
// CHECK:         %[[T1:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = ".tmp"}
// CHECK:         hlfir.assign %[[A]] to %[[T1]]#0 temporary_lhs
// CHECK:         %[[FALSE1:.*]] = arith.constant false
// CHECK:         fir.result %[[T1]]#0, %[[FALSE1]] : !fir.ref<!fir.type<_Tt{x:i32}>>, i1
// CHECK:       } else {
// CHECK:         %[[T2:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = ".tmp"}
// CHECK:         hlfir.assign %[[B]] to %[[T2]]#0 temporary_lhs
// CHECK:         %[[FALSE2:.*]] = arith.constant false
// CHECK:         fir.result %[[T2]]#0, %[[FALSE2]] : !fir.ref<!fir.type<_Tt{x:i32}>>, i1
// CHECK:       }

// --- Test 2: Scalar polymorphic (class) ---
// Both branches yield a !fir.class<!fir.type<T>> variable. A non-move as_expr
// allocates a polymorphic temp via the runtime, copies into it, and returns
// mustFree=true.
func.func @test_scalar_polymorphic(%cond: i1, %a: !fir.class<!fir.type<_Tt{x:i32}>>, %b: !fir.class<!fir.type<_Tt{x:i32}>>) {
  %0 = hlfir.conditional %cond : (i1) -> !hlfir.expr<!fir.type<_Tt{x:i32}>?> {
    hlfir.yield %a : !fir.class<!fir.type<_Tt{x:i32}>>
  } else {
    hlfir.yield %b : !fir.class<!fir.type<_Tt{x:i32}>>
  }
  hlfir.destroy %0 : !hlfir.expr<!fir.type<_Tt{x:i32}>?>
  return
}
// CHECK-LABEL: func.func @test_scalar_polymorphic(
// CHECK-SAME:    %[[COND:.*]]: i1, %[[A:.*]]: !fir.class<!fir.type<_Tt{x:i32}>>, %[[B:.*]]: !fir.class<!fir.type<_Tt{x:i32}>>)
// CHECK:       %[[IF:.*]]:2 = fir.if %[[COND]] -> (!fir.class<!fir.type<_Tt{x:i32}>>, i1) {
// CHECK:         fir.embox %{{.*}} source_box %[[A]]
// CHECK:         fir.call @_FortranAAllocatableAllocate
// CHECK:         %[[T1:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = ".tmp"}
// CHECK:         hlfir.assign %[[A]] to %[[T1]]#0 temporary_lhs
// CHECK:         %[[TRUE1:.*]] = arith.constant true
// CHECK:         fir.result %[[T1]]#0, %[[TRUE1]] : !fir.class<!fir.type<_Tt{x:i32}>>, i1
// CHECK:       } else {
// CHECK:         fir.embox %{{.*}} source_box %[[B]]
// CHECK:         fir.call @_FortranAAllocatableAllocate
// CHECK:         %[[T2:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = ".tmp"}
// CHECK:         hlfir.assign %[[B]] to %[[T2]]#0 temporary_lhs
// CHECK:         %[[TRUE2:.*]] = arith.constant true
// CHECK:         fir.result %[[T2]]#0, %[[TRUE2]] : !fir.class<!fir.type<_Tt{x:i32}>>, i1
// CHECK:       }

// --- Test 3: Polymorphic function results with cleanup region ---
// The branch body contains the call + declare; cleanup ops (Destroy + freemem)
// are in the yield's cleanup region. Because that cleanup would destroy the
// yielded storage, ConditionalOp must use non-move as_expr semantics here so
// AssociateOpConversion makes a copy that survives the replayed cleanup.
func.func @test_poly_func_result_cleanup(%cond: i1) {
  %alloca = fir.alloca !fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>> {bindc_name = ".result"}
  %0 = hlfir.conditional %cond : (i1) -> !hlfir.expr<!fir.type<_Tt{x:i32}>?> {
    %1 = fir.call @_QPmake_poly() fastmath<contract> : () -> !fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>
    fir.save_result %1 to %alloca : !fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>, !fir.ref<!fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>>
    %2 = fir.load %alloca : !fir.ref<!fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>>
    %3:2 = hlfir.declare %2 {uniq_name = ".tmp.func_result"} : (!fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>) -> (!fir.class<!fir.type<_Tt{x:i32}>>, !fir.class<!fir.type<_Tt{x:i32}>>)
    hlfir.yield %3#0 : !fir.class<!fir.type<_Tt{x:i32}>> cleanup {
      %4 = fir.load %alloca : !fir.ref<!fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>>
      %5 = fir.convert %4 : (!fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>) -> !fir.box<none>
      fir.call @_FortranADestroy(%5) fastmath<contract> : (!fir.box<none>) -> ()
      %6 = fir.load %alloca : !fir.ref<!fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>>
      %7 = fir.box_addr %6 : (!fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>) -> !fir.heap<!fir.type<_Tt{x:i32}>>
      %8 = fir.convert %7 : (!fir.heap<!fir.type<_Tt{x:i32}>>) -> i64
      %c0 = arith.constant 0 : i64
      %9 = arith.cmpi ne, %8, %c0 : i64
      fir.if %9 {
        fir.freemem %7 : !fir.heap<!fir.type<_Tt{x:i32}>>
      }
    }
  } else {
    %1 = fir.call @_QPmake_poly() fastmath<contract> : () -> !fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>
    fir.save_result %1 to %alloca : !fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>, !fir.ref<!fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>>
    %2 = fir.load %alloca : !fir.ref<!fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>>
    %3:2 = hlfir.declare %2 {uniq_name = ".tmp.func_result"} : (!fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>) -> (!fir.class<!fir.type<_Tt{x:i32}>>, !fir.class<!fir.type<_Tt{x:i32}>>)
    hlfir.yield %3#0 : !fir.class<!fir.type<_Tt{x:i32}>> cleanup {
      %4 = fir.load %alloca : !fir.ref<!fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>>
      %5 = fir.convert %4 : (!fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>) -> !fir.box<none>
      fir.call @_FortranADestroy(%5) fastmath<contract> : (!fir.box<none>) -> ()
      %6 = fir.load %alloca : !fir.ref<!fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>>
      %7 = fir.box_addr %6 : (!fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>) -> !fir.heap<!fir.type<_Tt{x:i32}>>
      %8 = fir.convert %7 : (!fir.heap<!fir.type<_Tt{x:i32}>>) -> i64
      %c0 = arith.constant 0 : i64
      %9 = arith.cmpi ne, %8, %c0 : i64
      fir.if %9 {
        fir.freemem %7 : !fir.heap<!fir.type<_Tt{x:i32}>>
      }
    }
  }
  hlfir.destroy %0 : !hlfir.expr<!fir.type<_Tt{x:i32}>?>
  return
}
// CHECK-LABEL: func.func @test_poly_func_result_cleanup(
// Verify in the then branch: declare the func result, copy into a new temp
// (because cleanup would free the original buffer), then replay cleanup ops,
// then return the copy with mustFree=true.
// CHECK:       fir.if %{{.*}} -> (!fir.class<!fir.type<_Tt{x:i32}>>, i1) {
// CHECK:         %[[DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = ".tmp.func_result"}
// CHECK:         %[[COPY:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = ".tmp"}
// CHECK:         hlfir.assign %[[DECL]]#0 to %[[COPY]]#0 temporary_lhs
// CHECK:         %[[TRUE:.*]] = arith.constant true
// CHECK:         fir.call @_FortranADestroy
// CHECK:         fir.if
// CHECK:           fir.freemem
// CHECK:         }
// CHECK:         fir.result %[[COPY]]#0, %[[TRUE]] : !fir.class<!fir.type<_Tt{x:i32}>>, i1

// --- Test 4: Array with static extents ---
// ExprType is !hlfir.expr<?xi32>, mold has static shape (10xi32).
// A non-move as_expr allocates a heap temp and copies into it (mustFree=true);
// castTempToResultType then converts the static-extent ref to a dynamic-extent
// box via fir.convert + fir.embox.
func.func @test_array_static(%cond: i1, %a: !fir.ref<!fir.array<10xi32>>, %b: !fir.ref<!fir.array<10xi32>>) {
  %0 = hlfir.conditional %cond : (i1) -> !hlfir.expr<?xi32> {
    hlfir.yield %a : !fir.ref<!fir.array<10xi32>>
  } else {
    hlfir.yield %b : !fir.ref<!fir.array<10xi32>>
  }
  hlfir.destroy %0 : !hlfir.expr<?xi32>
  return
}
// CHECK-LABEL: func.func @test_array_static(
// CHECK-SAME:    %[[COND:.*]]: i1, %[[A:.*]]: !fir.ref<!fir.array<10xi32>>, %[[B:.*]]: !fir.ref<!fir.array<10xi32>>)
// CHECK:       %[[IF:.*]]:2 = fir.if %[[COND]] -> (!fir.box<!fir.array<?xi32>>, i1) {
// Copy into a heap temp, then embox to match the box result type:
// CHECK:         %[[MEM1:.*]] = fir.allocmem !fir.array<10xi32>
// CHECK:         %[[DECL1:.*]]:2 = hlfir.declare %[[MEM1]]
// CHECK:         hlfir.assign %[[A]] to %[[DECL1]]#0 temporary_lhs
// CHECK:         %[[TRUE1:.*]] = arith.constant true
// CHECK:         %[[CVT1:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<?xi32>>
// CHECK:         %[[BOX1:.*]] = fir.embox %[[CVT1]](%{{.*}}) : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
// CHECK:         fir.result %[[BOX1]], %[[TRUE1]] : !fir.box<!fir.array<?xi32>>, i1
// CHECK:       } else {
// CHECK:         %[[MEM2:.*]] = fir.allocmem !fir.array<10xi32>
// CHECK:         %[[DECL2:.*]]:2 = hlfir.declare %[[MEM2]]
// CHECK:         hlfir.assign %[[B]] to %[[DECL2]]#0 temporary_lhs
// CHECK:         %[[TRUE2:.*]] = arith.constant true
// CHECK:         %[[CVT2:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<?xi32>>
// CHECK:         %[[BOX2:.*]] = fir.embox %[[CVT2]](%{{.*}}) : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
// CHECK:         fir.result %[[BOX2]], %[[TRUE2]] : !fir.box<!fir.array<?xi32>>, i1
// CHECK:       }

// --- Test 5: Scalar character (static length) ---
// For static-length characters, tempBaseType is !fir.ref<!fir.char<1,N>>.
// A non-move as_expr copies the yielded variable into a stack temp
// (mustFree=false).
func.func @test_scalar_char(%cond: i1, %a: !fir.ref<!fir.char<1,20>>, %b: !fir.ref<!fir.char<1,20>>) {
  %0 = hlfir.conditional %cond : (i1) -> !hlfir.expr<!fir.char<1,20>> {
    hlfir.yield %a : !fir.ref<!fir.char<1,20>>
  } else {
    hlfir.yield %b : !fir.ref<!fir.char<1,20>>
  }
  hlfir.destroy %0 : !hlfir.expr<!fir.char<1,20>>
  return
}
// CHECK-LABEL: func.func @test_scalar_char(
// CHECK-SAME:    %[[COND:.*]]: i1, %[[A:.*]]: !fir.ref<!fir.char<1,20>>, %[[B:.*]]: !fir.ref<!fir.char<1,20>>)
// CHECK:       %[[IF:.*]]:2 = fir.if %[[COND]] -> (!fir.ref<!fir.char<1,20>>, i1) {
// CHECK:         %[[T1:.*]]:2 = hlfir.declare %{{.*}} typeparams %{{.*}} {uniq_name = ".tmp"}
// CHECK:         hlfir.assign %[[A]] to %[[T1]]#0 temporary_lhs
// CHECK:         %[[FALSE1:.*]] = arith.constant false
// CHECK:         fir.result %[[T1]]#0, %[[FALSE1]] : !fir.ref<!fir.char<1,20>>, i1
// CHECK:       } else {
// CHECK:         %[[T2:.*]]:2 = hlfir.declare %{{.*}} typeparams %{{.*}} {uniq_name = ".tmp"}
// CHECK:         hlfir.assign %[[B]] to %[[T2]]#0 temporary_lhs
// CHECK:         %[[FALSE2:.*]] = arith.constant false
// CHECK:         fir.result %[[T2]]#0, %[[FALSE2]] : !fir.ref<!fir.char<1,20>>, i1
// CHECK:       }

// --- Test 6: Forwarding (as_expr with move, type matches, inside region) ---
// When hlfir.as_expr has move=true and var type == tempBaseType,
// forward directly without creating an extra temp.
func.func @test_forwarding(%cond: i1) {
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %0 = hlfir.conditional %cond : (i1) -> !hlfir.expr<?xi32> {
    %mem1 = fir.allocmem !fir.array<?xi32>, %c10 {uniq_name = ".tmp"}
    %decl1:2 = hlfir.declare %mem1(%shape) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
    %true = arith.constant true
    %expr1 = hlfir.as_expr %decl1#0 move %true : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<?xi32>
    hlfir.yield %expr1 : !hlfir.expr<?xi32>
  } else {
    %mem2 = fir.allocmem !fir.array<?xi32>, %c10 {uniq_name = ".tmp"}
    %decl2:2 = hlfir.declare %mem2(%shape) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
    %true = arith.constant true
    %expr2 = hlfir.as_expr %decl2#0 move %true : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<?xi32>
    hlfir.yield %expr2 : !hlfir.expr<?xi32>
  }
  hlfir.destroy %0 : !hlfir.expr<?xi32>
  return
}
// CHECK-LABEL: func.func @test_forwarding(
// CHECK:       %[[IF:.*]]:2 = fir.if %{{.*}} -> (!fir.box<!fir.array<?xi32>>, i1) {
// No temp allocation or assign — directly forward the var and mustFree:
// CHECK:         %[[MEM1:.*]] = fir.allocmem !fir.array<?xi32>
// CHECK:         %[[DECL1:.*]]:2 = hlfir.declare %[[MEM1]]
// CHECK-NOT:     hlfir.assign
// CHECK:         %[[TRUE1:.*]] = arith.constant true
// CHECK:         fir.result %[[DECL1]]#0, %[[TRUE1]] : !fir.box<!fir.array<?xi32>>, i1
// CHECK:       } else {
// CHECK:         %[[MEM2:.*]] = fir.allocmem !fir.array<?xi32>
// CHECK:         %[[DECL2:.*]]:2 = hlfir.declare %[[MEM2]]
// CHECK-NOT:     hlfir.assign
// CHECK:         %[[TRUE2:.*]] = arith.constant true
// CHECK:         fir.result %[[DECL2]]#0, %[[TRUE2]] : !fir.box<!fir.array<?xi32>>, i1
// CHECK:       }

// --- Test 7: Character array with static length ---
// tempBaseType is !fir.box<!fir.array<?x!fir.char<1,15>>>. A non-move as_expr
// allocates a heap temp and copies into it (mustFree=true); castTempToResultType
// then converts + emboxes.
func.func @test_char_array_static_len(%cond: i1, %a: !fir.ref<!fir.array<3x!fir.char<1,15>>>, %b: !fir.ref<!fir.array<3x!fir.char<1,15>>>) {
  %0 = hlfir.conditional %cond : (i1) -> !hlfir.expr<?x!fir.char<1,15>> {
    hlfir.yield %a : !fir.ref<!fir.array<3x!fir.char<1,15>>>
  } else {
    hlfir.yield %b : !fir.ref<!fir.array<3x!fir.char<1,15>>>
  }
  hlfir.destroy %0 : !hlfir.expr<?x!fir.char<1,15>>
  return
}
// CHECK-LABEL: func.func @test_char_array_static_len(
// CHECK-SAME:    %[[COND:.*]]: i1, %[[A:.*]]: !fir.ref<!fir.array<3x!fir.char<1,15>>>, %[[B:.*]]: !fir.ref<!fir.array<3x!fir.char<1,15>>>)
// CHECK:       %[[IF:.*]]:2 = fir.if %[[COND]] -> (!fir.box<!fir.array<?x!fir.char<1,15>>>, i1) {
// Copy into a heap temp, then convert + embox:
// CHECK:         %[[MEM1:.*]] = fir.allocmem !fir.array<3x!fir.char<1,15>>
// CHECK:         %[[DECL1:.*]]:2 = hlfir.declare %[[MEM1]]
// CHECK:         hlfir.assign %[[A]] to %[[DECL1]]#0 temporary_lhs
// CHECK:         %[[TRUE1:.*]] = arith.constant true
// CHECK:         %[[CVT1:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.array<3x!fir.char<1,15>>>) -> !fir.ref<!fir.array<?x!fir.char<1,15>>>
// CHECK:         %[[BOX1:.*]] = fir.embox %[[CVT1]](%{{.*}}) : (!fir.ref<!fir.array<?x!fir.char<1,15>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.char<1,15>>>
// CHECK:         fir.result %[[BOX1]], %[[TRUE1]] : !fir.box<!fir.array<?x!fir.char<1,15>>>, i1
// CHECK:       } else {
// CHECK:         %[[MEM2:.*]] = fir.allocmem !fir.array<3x!fir.char<1,15>>
// CHECK:         %[[DECL2:.*]]:2 = hlfir.declare %[[MEM2]]
// CHECK:         hlfir.assign %[[B]] to %[[DECL2]]#0 temporary_lhs
// CHECK:         %[[TRUE2:.*]] = arith.constant true
// CHECK:         %[[CVT2:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.array<3x!fir.char<1,15>>>) -> !fir.ref<!fir.array<?x!fir.char<1,15>>>
// CHECK:         %[[BOX2:.*]] = fir.embox %[[CVT2]](%{{.*}}) : (!fir.ref<!fir.array<?x!fir.char<1,15>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.char<1,15>>>
// CHECK:         fir.result %[[BOX2]], %[[TRUE2]] : !fir.box<!fir.array<?x!fir.char<1,15>>>, i1
// CHECK:       }

// --- Test 8: Scalar character (dynamic length) ---
// tempBaseType is !fir.boxchar<1>. A non-move as_expr copies the yielded
// variable into a stack temp (mustFree=false).
func.func @test_scalar_char_dynamic(%cond: i1, %a: !fir.boxchar<1>, %b: !fir.boxchar<1>) {
  %0 = hlfir.conditional %cond : (i1) -> !hlfir.expr<!fir.char<1,?>> {
    hlfir.yield %a : !fir.boxchar<1>
  } else {
    hlfir.yield %b : !fir.boxchar<1>
  }
  hlfir.destroy %0 : !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL: func.func @test_scalar_char_dynamic(
// CHECK-SAME:    %[[COND:.*]]: i1, %[[A:.*]]: !fir.boxchar<1>, %[[B:.*]]: !fir.boxchar<1>)
// CHECK:       %[[IF:.*]]:2 = fir.if %[[COND]] -> (!fir.boxchar<1>, i1) {
// CHECK:         %[[T1:.*]]:2 = hlfir.declare %{{.*}} typeparams %{{.*}} {uniq_name = ".tmp"}
// CHECK:         hlfir.assign %[[A]] to %[[T1]]#0 temporary_lhs
// CHECK:         %[[FALSE1:.*]] = arith.constant false
// CHECK:         fir.result %[[T1]]#0, %[[FALSE1]] : !fir.boxchar<1>, i1
// CHECK:       } else {
// CHECK:         %[[T2:.*]]:2 = hlfir.declare %{{.*}} typeparams %{{.*}} {uniq_name = ".tmp"}
// CHECK:         hlfir.assign %[[B]] to %[[T2]]#0 temporary_lhs
// CHECK:         %[[FALSE2:.*]] = arith.constant false
// CHECK:         fir.result %[[T2]]#0, %[[FALSE2]] : !fir.boxchar<1>, i1
// CHECK:       }

// --- Test 9: Asymmetric branches (non-polymorphic + polymorphic) ---
// One branch yields !fir.ref (non-polymorphic), the other !fir.class
// (polymorphic). ExprType is polymorphic, so tempBaseType is !fir.class.
// Each branch copies its variable into a temp; the non-poly branch then emboxes
// its ref temp to class.
func.func @test_asymmetric_poly(%cond: i1, %a: !fir.ref<!fir.type<_Tt{x:i32}>>, %b: !fir.class<!fir.type<_Tt{x:i32}>>) {
  %0 = hlfir.conditional %cond : (i1) -> !hlfir.expr<!fir.type<_Tt{x:i32}>?> {
    hlfir.yield %a : !fir.ref<!fir.type<_Tt{x:i32}>>
  } else {
    hlfir.yield %b : !fir.class<!fir.type<_Tt{x:i32}>>
  }
  hlfir.destroy %0 : !hlfir.expr<!fir.type<_Tt{x:i32}>?>
  return
}
// CHECK-LABEL: func.func @test_asymmetric_poly(
// CHECK-SAME:    %[[COND:.*]]: i1, %[[A:.*]]: !fir.ref<!fir.type<_Tt{x:i32}>>, %[[B:.*]]: !fir.class<!fir.type<_Tt{x:i32}>>)
// CHECK:       %[[IF:.*]]:2 = fir.if %[[COND]] -> (!fir.class<!fir.type<_Tt{x:i32}>>, i1) {
// Non-poly branch: copy into a stack temp, then embox ref → class:
// CHECK:         %[[T1:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = ".tmp"}
// CHECK:         hlfir.assign %[[A]] to %[[T1]]#0 temporary_lhs
// CHECK:         %[[FALSE:.*]] = arith.constant false
// CHECK:         %[[BOX:.*]] = fir.embox %[[T1]]#0 : (!fir.ref<!fir.type<_Tt{x:i32}>>) -> !fir.class<!fir.type<_Tt{x:i32}>>
// CHECK:         fir.result %[[BOX]], %[[FALSE]] : !fir.class<!fir.type<_Tt{x:i32}>>, i1
// CHECK:       } else {
// Poly branch: runtime-allocate a polymorphic temp and copy into it:
// CHECK:         fir.call @_FortranAAllocatableAllocate
// CHECK:         %[[T2:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = ".tmp"}
// CHECK:         hlfir.assign %[[B]] to %[[T2]]#0 temporary_lhs
// CHECK:         %[[TRUE:.*]] = arith.constant true
// CHECK:         fir.result %[[T2]]#0, %[[TRUE]] : !fir.class<!fir.type<_Tt{x:i32}>>, i1
// CHECK:       }

// Helper declarations for test 3.
func.func private @_QPmake_poly() -> !fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>
func.func private @_FortranADestroy(!fir.box<none>) -> ()

// --- Test 10: Forwarding with non-empty cleanup region ---
// When forwarding (as_expr move, type matches), the cleanup region may
// contain ops beyond the destroy of the forwarded entity — e.g.,
// freeing argument temporaries after a function call.
// The forwarding path must replay those cleanups while skipping only
// the destroy whose operand is the forwarded entity.
// Uses FIR-level cleanup ops (fir.freemem) so they survive lowering and
// are visible in the CHECK output.
func.func @test_forwarding_with_cleanup(%cond: i1, %arg: !fir.boxchar<1>) {
  %result_alloca = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {bindc_name = ".result"}
  %0 = hlfir.conditional %cond : (i1) -> !hlfir.expr<!fir.char<1,?>> {
    // Simulate a heap-allocated argument temporary.
    %c10 = arith.constant 10 : index
    %heap = fir.allocmem !fir.char<1,?>(%c10 : index) {uniq_name = ".tmp.arg"}
    // Simulate function call returning allocatable character result.
    %call = fir.call @_QPfunc_alloc_char(%arg) : (!fir.boxchar<1>) -> !fir.box<!fir.heap<!fir.char<1,?>>>
    fir.save_result %call to %result_alloca : !fir.box<!fir.heap<!fir.char<1,?>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
    %loaded = fir.load %result_alloca : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
    %addr = fir.box_addr %loaded : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
    %len = fir.box_elesize %loaded : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> index
    %boxchar = fir.emboxchar %addr, %len : (!fir.heap<!fir.char<1,?>>, index) -> !fir.boxchar<1>
    %true = arith.constant true
    %expr = hlfir.as_expr %boxchar move %true : (!fir.boxchar<1>, i1) -> !hlfir.expr<!fir.char<1,?>>
    hlfir.yield %expr : !hlfir.expr<!fir.char<1,?>> cleanup {
      hlfir.destroy %expr : !hlfir.expr<!fir.char<1,?>>
      fir.freemem %heap : !fir.heap<!fir.char<1,?>>
    }
  } else {
    hlfir.yield %arg : !fir.boxchar<1>
  }
  hlfir.assign %0 to %arg : !hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>
  hlfir.destroy %0 : !hlfir.expr<!fir.char<1,?>>
  return
}
func.func private @_QPfunc_alloc_char(!fir.boxchar<1>) -> !fir.box<!fir.heap<!fir.char<1,?>>>
// CHECK-LABEL: func.func @test_forwarding_with_cleanup(
// CHECK-SAME:    %[[COND:.*]]: i1, %[[ARG:.*]]: !fir.boxchar<1>)
// CHECK:       %[[IF:.*]]:2 = fir.if %[[COND]] -> (!fir.boxchar<1>, i1) {
// Forwarding: var and mustFree forwarded directly, no temp copy:
// CHECK:         %[[HEAP:.*]] = fir.allocmem !fir.char<1,?>
// CHECK:         fir.call @_QPfunc_alloc_char
// CHECK:         %[[BOXCHAR:.*]] = fir.emboxchar
// CHECK:         %[[TRUE:.*]] = arith.constant true
// Cleanup replayed — arg temp freed:
// CHECK:         fir.freemem %[[HEAP]]
// CHECK:         fir.result %[[BOXCHAR]], %[[TRUE]] : !fir.boxchar<1>, i1
// CHECK:       } else {
// Non-forwarding: variable copied into a stack temp (mustFree=false):
// CHECK:         %[[T:.*]]:2 = hlfir.declare %{{.*}} typeparams %{{.*}} {uniq_name = ".tmp"}
// CHECK:         hlfir.assign %[[ARG]] to %[[T]]#0 temporary_lhs
// CHECK:         %[[FALSE:.*]] = arith.constant false
// CHECK:         fir.result %[[T]]#0, %[[FALSE]] : !fir.boxchar<1>, i1
// CHECK:       }

// --- Test 11: Array expr from hlfir.elemental (buffer forwarding) ---
// When a branch yields an hlfir.expr produced by hlfir.elemental, the
// elemental's heap buffer is forwarded directly via hlfir.associate without
// an extra copy.  The other branch yields a variable (exercises the
// convert + embox path for castTempToResultType).
func.func @test_elemental_forwarding(%cond: i1, %n: index, %a: !fir.ref<!fir.array<10xi32>>) {
  %shape = fir.shape %n : (index) -> !fir.shape<1>
  %0 = hlfir.conditional %cond : (i1) -> !hlfir.expr<?xi32> {
    %elem = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
    ^bb0(%i: index):
      %c42 = arith.constant 42 : i32
      hlfir.yield_element %c42 : i32
    }
    hlfir.yield %elem : !hlfir.expr<?xi32>
  } else {
    hlfir.yield %a : !fir.ref<!fir.array<10xi32>>
  }
  hlfir.destroy %0 : !hlfir.expr<?xi32>
  return
}
// CHECK-LABEL: func.func @test_elemental_forwarding(
// CHECK-SAME:    %[[COND:.*]]: i1, %[[N:.*]]: index, %[[A:.*]]: !fir.ref<!fir.array<10xi32>>)
// CHECK:       %[[IF:.*]]:2 = fir.if %[[COND]] -> (!fir.box<!fir.array<?xi32>>, i1) {
// Elemental bufferized: allocmem + loop fills the temp:
// CHECK:         %[[ELEM_BUF:.*]] = fir.allocmem !fir.array<?xi32>, %[[N]]
// CHECK:         %[[ELEM_DECL:.*]]:2 = hlfir.declare %[[ELEM_BUF]](%{{.*}})
// CHECK:         %[[TRUE1:.*]] = arith.constant true
// CHECK:         fir.do_loop
// Buffer forwarded directly (no extra copy after loop):
// CHECK:         fir.result %[[ELEM_DECL]]#0, %[[TRUE1]] : !fir.box<!fir.array<?xi32>>, i1
// CHECK:       } else {
// Variable yield: copied into a heap temp (mustFree=true), then converted + emboxed:
// CHECK:         %[[MEM:.*]] = fir.allocmem !fir.array<10xi32>
// CHECK:         %[[DECL:.*]]:2 = hlfir.declare %[[MEM]]
// CHECK:         hlfir.assign %[[A]] to %[[DECL]]#0 temporary_lhs
// CHECK:         %[[TRUE2:.*]] = arith.constant true
// CHECK:         %[[CVT:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<?xi32>>
// CHECK:         %[[BOX:.*]] = fir.embox %[[CVT]](%{{.*}}) : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
// CHECK:         fir.result %[[BOX]], %[[TRUE2]] : !fir.box<!fir.array<?xi32>>, i1
// CHECK:       }

// --- Test 12: Nested conditional (exercises setHasBoundedRewriteRecursion) ---
// An inner hlfir.conditional inside an outer branch verifies that the dialect
// conversion framework correctly re-applies ConditionalOpConversion to the
// cloned inner op via bounded rewrite recursion.
func.func @test_nested(%cond1: i1, %cond2: i1, %a: !fir.ref<!fir.type<_Tt{x:i32}>>, %b: !fir.ref<!fir.type<_Tt{x:i32}>>, %c: !fir.ref<!fir.type<_Tt{x:i32}>>) {
  %0 = hlfir.conditional %cond1 : (i1) -> !hlfir.expr<!fir.type<_Tt{x:i32}>> {
    %1 = hlfir.conditional %cond2 : (i1) -> !hlfir.expr<!fir.type<_Tt{x:i32}>> {
      hlfir.yield %a : !fir.ref<!fir.type<_Tt{x:i32}>>
    } else {
      hlfir.yield %b : !fir.ref<!fir.type<_Tt{x:i32}>>
    }
    hlfir.yield %1 : !hlfir.expr<!fir.type<_Tt{x:i32}>>
  } else {
    hlfir.yield %c : !fir.ref<!fir.type<_Tt{x:i32}>>
  }
  hlfir.destroy %0 : !hlfir.expr<!fir.type<_Tt{x:i32}>>
  return
}
// CHECK-LABEL: func.func @test_nested(
// CHECK-SAME:    %[[COND1:.*]]: i1, %[[COND2:.*]]: i1, %[[A:.*]]: !fir.ref<!fir.type<_Tt{x:i32}>>, %[[B:.*]]: !fir.ref<!fir.type<_Tt{x:i32}>>, %[[C:.*]]: !fir.ref<!fir.type<_Tt{x:i32}>>)
// Outer fir.if:
// CHECK:       %[[OUTER:.*]]:2 = fir.if %[[COND1]] -> (!fir.ref<!fir.type<_Tt{x:i32}>>, i1) {
// Inner fir.if (nested — proves bounded rewrite recursion works):
// CHECK:         %[[INNER:.*]]:2 = fir.if %[[COND2]] -> (!fir.ref<!fir.type<_Tt{x:i32}>>, i1) {
// CHECK:           %[[IT:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = ".tmp"}
// CHECK:           hlfir.assign %[[A]] to %[[IT]]#0 temporary_lhs
// CHECK:           fir.result %[[IT]]#0, %{{.*}} : !fir.ref<!fir.type<_Tt{x:i32}>>, i1
// CHECK:         } else {
// CHECK:           %[[IE:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = ".tmp"}
// CHECK:           hlfir.assign %[[B]] to %[[IE]]#0 temporary_lhs
// CHECK:           fir.result %[[IE]]#0, %{{.*}} : !fir.ref<!fir.type<_Tt{x:i32}>>, i1
// CHECK:         }
// Inner results forwarded to outer fir.result:
// CHECK:         fir.result %[[INNER]]#0, %[[INNER]]#1 : !fir.ref<!fir.type<_Tt{x:i32}>>, i1
// CHECK:       } else {
// CHECK:         %[[OE:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = ".tmp"}
// CHECK:         hlfir.assign %[[C]] to %[[OE]]#0 temporary_lhs
// CHECK:         fir.result %[[OE]]#0, %{{.*}} : !fir.ref<!fir.type<_Tt{x:i32}>>, i1
// CHECK:       }
