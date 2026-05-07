// Test hlfir.conditional bufferization (ConditionalOpConversion).
// RUN: fir-opt --bufferize-hlfir %s | FileCheck %s

// --- Test 1: Scalar non-polymorphic derived type ---
// Both branches yield !fir.ref<!fir.type<T>>. No cast needed.
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
// CHECK:         hlfir.assign %[[A]] to %{{.*}}#0 temporary_lhs
// CHECK:         %[[FALSE1:.*]] = arith.constant false
// CHECK:         fir.result %{{.*}}#0, %[[FALSE1]] : !fir.ref<!fir.type<_Tt{x:i32}>>, i1
// CHECK:       } else {
// CHECK:         hlfir.assign %[[B]] to %{{.*}}#0 temporary_lhs
// CHECK:         %[[FALSE2:.*]] = arith.constant false
// CHECK:         fir.result %{{.*}}#0, %[[FALSE2]] : !fir.ref<!fir.type<_Tt{x:i32}>>, i1
// CHECK:       }

// --- Test 2: Scalar polymorphic (class) ---
// Both branches yield !fir.class<!fir.type<T>>. createTempFromMold produces a
// polymorphic allocatable temp via runtime allocation.
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
// CHECK:         hlfir.assign %[[A]] to %{{.*}}#0 temporary_lhs
// CHECK:         %[[TRUE1:.*]] = arith.constant true
// CHECK:         fir.result %{{.*}}, %[[TRUE1]] : !fir.class<!fir.type<_Tt{x:i32}>>, i1
// CHECK:       } else {
// CHECK:         fir.embox %{{.*}} source_box %[[B]]
// CHECK:         fir.call @_FortranAAllocatableAllocate
// CHECK:         hlfir.assign %[[B]] to %{{.*}}#0 temporary_lhs
// CHECK:         %[[TRUE2:.*]] = arith.constant true
// CHECK:         fir.result %{{.*}}, %[[TRUE2]] : !fir.class<!fir.type<_Tt{x:i32}>>, i1
// CHECK:       }

// --- Test 3: Polymorphic function results with deferred cleanups ---
// The branch body contains cleanup ops (Destroy + freemem) after the declare.
// Verify that assign happens BEFORE the cleanup (no use-after-free).
func.func @test_poly_func_result_cleanup(%cond: i1) {
  %alloca = fir.alloca !fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>> {bindc_name = ".result"}
  %0 = hlfir.conditional %cond : (i1) -> !hlfir.expr<!fir.type<_Tt{x:i32}>?> {
    %1 = fir.call @_QPmake_poly() fastmath<contract> : () -> !fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>
    fir.save_result %1 to %alloca : !fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>, !fir.ref<!fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>>
    %2 = fir.load %alloca : !fir.ref<!fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>>
    %3:2 = hlfir.declare %2 {uniq_name = ".tmp.func_result"} : (!fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>) -> (!fir.class<!fir.type<_Tt{x:i32}>>, !fir.class<!fir.type<_Tt{x:i32}>>)
    // Cleanup: Destroy + freemem (must run AFTER assign in bufferized output)
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
    hlfir.yield %3#0 : !fir.class<!fir.type<_Tt{x:i32}>>
  } else {
    %1 = fir.call @_QPmake_poly() fastmath<contract> : () -> !fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>
    fir.save_result %1 to %alloca : !fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>, !fir.ref<!fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>>
    %2 = fir.load %alloca : !fir.ref<!fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>>
    %3:2 = hlfir.declare %2 {uniq_name = ".tmp.func_result"} : (!fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>) -> (!fir.class<!fir.type<_Tt{x:i32}>>, !fir.class<!fir.type<_Tt{x:i32}>>)
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
    hlfir.yield %3#0 : !fir.class<!fir.type<_Tt{x:i32}>>
  }
  hlfir.destroy %0 : !hlfir.expr<!fir.type<_Tt{x:i32}>?>
  return
}
// CHECK-LABEL: func.func @test_poly_func_result_cleanup(
// Verify in the then branch: declare, then createTempFromMold + assign, THEN destroy/freemem.
// CHECK:       fir.if %{{.*}} -> (!fir.class<!fir.type<_Tt{x:i32}>>, i1) {
// CHECK:         %[[DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = ".tmp.func_result"}
// CHECK:         fir.embox %{{.*}} source_box %[[DECL]]#0
// CHECK:         fir.call @_FortranAAllocatableAllocate
// CHECK:         hlfir.assign %[[DECL]]#0 to %{{.*}} temporary_lhs
// CHECK:         fir.call @_FortranADestroy
// CHECK:         fir.if
// CHECK:           fir.freemem
// CHECK:         }
// CHECK:         fir.result

// --- Test 4: Array with static extents ---
// ExprType is !hlfir.expr<?xi32>, mold has static shape (10xi32).
// createTempFromMold produces !fir.heap<!fir.array<10xi32>>, needs embox.
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
// CHECK:         %[[ALLOC:.*]] = fir.allocmem !fir.array<10xi32>
// CHECK:         %[[DECL:.*]]:2 = hlfir.declare %[[ALLOC]]
// CHECK:         hlfir.assign %[[A]] to %[[DECL]]#0 temporary_lhs
// CHECK:         %[[BOX:.*]] = fir.embox %[[DECL]]#0(%{{.*}}) : (!fir.heap<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
// CHECK:         %[[TRUE:.*]] = arith.constant true
// CHECK:         fir.result %[[BOX]], %[[TRUE]] : !fir.box<!fir.array<?xi32>>, i1
// CHECK:       } else {
// CHECK:         %[[ALLOC2:.*]] = fir.allocmem !fir.array<10xi32>
// CHECK:         %[[DECL2:.*]]:2 = hlfir.declare %[[ALLOC2]]
// CHECK:         hlfir.assign %[[B]] to %[[DECL2]]#0 temporary_lhs
// CHECK:         %[[BOX2:.*]] = fir.embox %[[DECL2]]#0(%{{.*}}) : (!fir.heap<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
// CHECK:         %[[TRUE2:.*]] = arith.constant true
// CHECK:         fir.result %[[BOX2]], %[[TRUE2]] : !fir.box<!fir.array<?xi32>>, i1
// CHECK:       }

// --- Test 5: Scalar character (static length) ---
// For static-length characters, tempBaseType is !fir.ref<!fir.char<1,N>>,
// and createTempFromMold produces the same type. No cast needed.
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
// CHECK:         hlfir.assign %[[A]] to %{{.*}}#0 temporary_lhs
// CHECK:         %[[FALSE1:.*]] = arith.constant false
// CHECK:         fir.result %{{.*}}#0, %[[FALSE1]] : !fir.ref<!fir.char<1,20>>, i1
// CHECK:       } else {
// CHECK:         hlfir.assign %[[B]] to %{{.*}}#0 temporary_lhs
// CHECK:         %[[FALSE2:.*]] = arith.constant false
// CHECK:         fir.result %{{.*}}#0, %[[FALSE2]] : !fir.ref<!fir.char<1,20>>, i1
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
// tempBaseType is !fir.box<!fir.array<?x!fir.char<1,15>>>.
// fir.embox should NOT pass length params (static length encoded in type).
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
// CHECK:         %[[ALLOC:.*]] = fir.allocmem !fir.array<3x!fir.char<1,15>>
// CHECK:         %[[DECL:.*]]:2 = hlfir.declare %[[ALLOC]]
// CHECK:         hlfir.assign %[[A]] to %[[DECL]]#0 temporary_lhs
// Verify embox has NO typeparams (no extra operands after shape):
// CHECK:         %[[BOX:.*]] = fir.embox %[[DECL]]#0(%{{.*}}) : (!fir.heap<!fir.array<3x!fir.char<1,15>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.char<1,15>>>
// CHECK:         %[[TRUE:.*]] = arith.constant true
// CHECK:         fir.result %[[BOX]], %[[TRUE]] : !fir.box<!fir.array<?x!fir.char<1,15>>>, i1
// CHECK:       } else {
// CHECK:         %[[ALLOC2:.*]] = fir.allocmem !fir.array<3x!fir.char<1,15>>
// CHECK:         %[[DECL2:.*]]:2 = hlfir.declare %[[ALLOC2]]
// CHECK:         hlfir.assign %[[B]] to %[[DECL2]]#0 temporary_lhs
// CHECK:         %[[BOX2:.*]] = fir.embox %[[DECL2]]#0(%{{.*}}) : (!fir.heap<!fir.array<3x!fir.char<1,15>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.char<1,15>>>
// CHECK:         %[[TRUE2:.*]] = arith.constant true
// CHECK:         fir.result %[[BOX2]], %[[TRUE2]] : !fir.box<!fir.array<?x!fir.char<1,15>>>, i1
// CHECK:       }

// --- Test 8: Scalar character (dynamic length) ---
// tempBaseType is !fir.boxchar<1>. createTempFromMold uses fir.unboxchar to
// extract the length and produces a stack temp with the same type, so no cast.
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
// Dynamic length extracted from mold and used for stack alloca:
// CHECK:         fir.unboxchar %[[A]]
// CHECK:         fir.alloca !fir.char<1,?>
// CHECK:         hlfir.assign %[[A]] to %{{.*}}#0 temporary_lhs
// CHECK:         %[[FALSE1:.*]] = arith.constant false
// CHECK:         fir.result %{{.*}}, %[[FALSE1]] : !fir.boxchar<1>, i1
// CHECK:       } else {
// CHECK:         fir.unboxchar %[[B]]
// CHECK:         fir.alloca !fir.char<1,?>
// CHECK:         hlfir.assign %[[B]] to %{{.*}}#0 temporary_lhs
// CHECK:         %[[FALSE2:.*]] = arith.constant false
// CHECK:         fir.result %{{.*}}, %[[FALSE2]] : !fir.boxchar<1>, i1
// CHECK:       }

// --- Test 9: Asymmetric branches (non-polymorphic + polymorphic) ---
// One branch yields !fir.ref (non-polymorphic), the other !fir.class
// (polymorphic). ExprType is polymorphic, so tempBaseType is !fir.class.
// The non-poly branch needs fir.embox to cast the ref temp → class.
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
// Non-poly branch: stack alloca, assign, then embox ref → class:
// CHECK:         hlfir.assign %[[A]] to %{{.*}}#0 temporary_lhs
// CHECK:         fir.embox %{{.*}} : (!fir.ref<!fir.type<_Tt{x:i32}>>) -> !fir.class<!fir.type<_Tt{x:i32}>>
// CHECK:         %[[FALSE:.*]] = arith.constant false
// CHECK:         fir.result %{{.*}}, %[[FALSE]] : !fir.class<!fir.type<_Tt{x:i32}>>, i1
// CHECK:       } else {
// Poly branch: runtime allocation + assign, no cast needed:
// CHECK:         fir.embox %{{.*}} source_box %[[B]]
// CHECK:         fir.call @_FortranAAllocatableAllocate
// CHECK:         hlfir.assign %[[B]] to %{{.*}}#0 temporary_lhs
// CHECK:         %[[TRUE:.*]] = arith.constant true
// CHECK:         fir.result %{{.*}}, %[[TRUE]] : !fir.class<!fir.type<_Tt{x:i32}>>, i1
// CHECK:       }

// Helper declarations for test 3.
func.func private @_QPmake_poly() -> !fir.class<!fir.heap<!fir.type<_Tt{x:i32}>>>
func.func private @_FortranADestroy(!fir.box<none>) -> ()
