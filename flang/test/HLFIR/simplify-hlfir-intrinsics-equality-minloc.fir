// RUN: fir-opt %s --simplify-hlfir-intrinsics | FileCheck %s

// Rank 1: Variable: A == %target
func.func @test_minloc_1d_equality_variable(%arg0: !hlfir.expr<?xi32>, %target: i32) -> !hlfir.expr<1xi32> {
  %shape = hlfir.shape_of %arg0 : (!hlfir.expr<?xi32>) -> !fir.shape<1>
  %mask = hlfir.elemental %shape : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%i: index):
    %val = hlfir.apply %arg0, %i : (!hlfir.expr<?xi32>, index) -> i32
    %cmp = arith.cmpi eq, %val, %target : i32
    %logical = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %logical : !fir.logical<4>
  }
  %res = hlfir.minloc %arg0 mask %mask : (!hlfir.expr<?xi32>, !hlfir.expr<?x!fir.logical<4>>) -> !hlfir.expr<1xi32>
  return %res : !hlfir.expr<1xi32>
}
// CHECK-LABEL: func.func @test_minloc_1d_equality_variable
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[TRUE:.*]] = arith.constant true
// CHECK-NOT: arith.constant 2147483647 : i32
// CHECK: fir.do_loop %[[IV:.*]] = {{.*}} iter_args(%[[LOC:.*]] = %[[C0]], %[[FIRST:.*]] = %[[TRUE]]) -> (i32, i1)
// CHECK-NOT: hlfir.apply {{.*}} !fir.logical<4>
// CHECK:   %[[VAL:.*]] = hlfir.apply %{{.*}}, %[[IV]]
// CHECK:   %[[EQ:.*]] = arith.cmpi eq, %[[VAL]], %{{.*}}
// CHECK:   %[[COND:.*]] = arith.andi %[[EQ]], %[[FIRST]]
// CHECK:   %[[IF_RES:.*]]:2 = fir.if %[[COND]] -> (i32, i1)

// Rank 2: Variable: A == %target
func.func @test_minloc_2d_equality_variable(%arg0: !hlfir.expr<?x?xi32>, %target: i32) -> !hlfir.expr<2xi32> {
  %shape = hlfir.shape_of %arg0 : (!hlfir.expr<?x?xi32>) -> !fir.shape<2>
  %mask = hlfir.elemental %shape : (!fir.shape<2>) -> !hlfir.expr<?x?x!fir.logical<4>> {
  ^bb0(%i: index, %j: index):
    %val = hlfir.apply %arg0, %i, %j : (!hlfir.expr<?x?xi32>, index, index) -> i32
    %cmp = arith.cmpi eq, %val, %target : i32
    %logical = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %logical : !fir.logical<4>
  }
  %res = hlfir.minloc %arg0 mask %mask : (!hlfir.expr<?x?xi32>, !hlfir.expr<?x?x!fir.logical<4>>) -> !hlfir.expr<2xi32>
  return %res : !hlfir.expr<2xi32>
}
// CHECK-LABEL: func.func @test_minloc_2d_equality_variable
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[TRUE:.*]] = arith.constant true
// CHECK: %[[RES_OUTER:.*]]:3 = fir.do_loop %[[IV1:.*]] = {{.*}} iter_args(%[[OUT1:.*]] = %[[C0]], %[[OUT2:.*]] = %[[C0]], %[[OUT3:.*]] = %[[TRUE]]) -> (i32, i32, i1)
// CHECK:   %[[RES_INNER:.*]]:3 = fir.do_loop %[[IV2:.*]] = {{.*}} iter_args(%[[IN1:.*]] = %[[OUT1]], %[[IN2:.*]] = %[[OUT2]], %[[IN3:.*]] = %[[OUT3]]) -> (i32, i32, i1)
// CHECK-NOT: hlfir.apply {{.*}} !fir.logical<4>
// CHECK:     %[[EQ:.*]] = arith.cmpi eq, {{.*}}
// CHECK:     %[[COND:.*]] = arith.andi %[[EQ]], %[[IN3]]
// CHECK:     %[[IF:.*]]:3 = fir.if %[[COND]] -> (i32, i32, i1)

// Rank 3: Variable: A == %target
func.func @test_minloc_3d_equality_variable(%arg0: !hlfir.expr<?x?x?xi32>, %target: i32) -> !hlfir.expr<3xi32> {
  %shape = hlfir.shape_of %arg0 : (!hlfir.expr<?x?x?xi32>) -> !fir.shape<3>
  %mask = hlfir.elemental %shape : (!fir.shape<3>) -> !hlfir.expr<?x?x?x!fir.logical<4>> {
  ^bb0(%i: index, %j: index, %k: index):
    %val = hlfir.apply %arg0, %i, %j, %k : (!hlfir.expr<?x?x?xi32>, index, index, index) -> i32
    %cmp = arith.cmpi eq, %val, %target : i32
    %logical = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %logical : !fir.logical<4>
  }
  %res = hlfir.minloc %arg0 mask %mask : (!hlfir.expr<?x?x?xi32>, !hlfir.expr<?x?x?x!fir.logical<4>>) -> !hlfir.expr<3xi32>
  return %res : !hlfir.expr<3xi32>
}
// CHECK-LABEL: func.func @test_minloc_3d_equality_variable
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[TRUE:.*]] = arith.constant true
// CHECK: %[[OUTER:.*]]:4 = fir.do_loop %[[IV1:.*]] = {{.*}} iter_args(%[[O1:.*]] = %[[C0]], %[[O2:.*]] = %[[C0]], %[[O3:.*]] = %[[C0]], %[[O4:.*]] = %[[TRUE]]) -> (i32, i32, i32, i1)
// CHECK:   %[[MIDDLE:.*]]:4 = fir.do_loop %[[IV2:.*]] = {{.*}} iter_args(%[[M1:.*]] = %[[O1]], %[[M2:.*]] = %[[O2]], %[[M3:.*]] = %[[O3]], %[[M4:.*]] = %[[O4]]) -> (i32, i32, i32, i1)
// CHECK:     %[[INNER:.*]]:4 = fir.do_loop %[[IV3:.*]] = {{.*}} iter_args(%[[I1:.*]] = %[[M1]], %[[I2:.*]] = %[[M2]], %[[I3:.*]] = %[[M3]], %[[I4:.*]] = %[[M4]]) -> (i32, i32, i32, i1)
// CHECK-NOT: hlfir.apply {{.*}} !fir.logical<4>
// CHECK:       %[[EQ:.*]] = arith.cmpi eq, {{.*}}
// CHECK:       %[[COND:.*]] = arith.andi %[[EQ]], %[[I4]]
// CHECK:       %[[IF:.*]]:4 = fir.if %[[COND]] -> (i32, i32, i32, i1)

// Rank 1: Constant: A == 42
func.func @test_minloc_1d_equality_constant(%arg0: !hlfir.expr<?xi32>) -> !hlfir.expr<1xi32> {
  %c42 = arith.constant 42 : i32
  %shape = hlfir.shape_of %arg0 : (!hlfir.expr<?xi32>) -> !fir.shape<1>
  %mask = hlfir.elemental %shape : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%i: index):
    %val = hlfir.apply %arg0, %i : (!hlfir.expr<?xi32>, index) -> i32
    %cmp = arith.cmpi eq, %val, %c42 : i32
    %logical = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %logical : !fir.logical<4>
  }
  %res = hlfir.minloc %arg0 mask %mask : (!hlfir.expr<?xi32>, !hlfir.expr<?x!fir.logical<4>>) -> !hlfir.expr<1xi32>
  return %res : !hlfir.expr<1xi32>
}
// CHECK-LABEL: func.func @test_minloc_1d_equality_constant
// CHECK-DAG: %[[C42:.*]] = arith.constant 42 : i32
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[TRUE:.*]] = arith.constant true

// CHECK: %[[RES:.*]]:2 = fir.do_loop %[[IV:.*]] = {{.*}} iter_args(%[[LOC:.*]] = %[[C0]], %[[FIRST:.*]] = %[[TRUE]]) -> (i32, i1)
// CHECK-NOT: hlfir.apply {{.*}} !fir.logical<4>
// CHECK:   %[[VAL:.*]] = hlfir.apply %{{.*}}, %[[IV]]
// CHECK:   %[[EQ:.*]] = arith.cmpi eq, %[[VAL]], %[[C42]]
// CHECK:   %[[COND:.*]] = arith.andi %[[EQ]], %[[FIRST]]
// CHECK:   %[[IF:.*]]:2 = fir.if %[[COND]] -> (i32, i1)

// Rank 2: Constant: A == 42
func.func @test_minloc_2d_equality_constant(%arg0: !hlfir.expr<?x?xi32>) -> !hlfir.expr<2xi32> {
  %c42 = arith.constant 42 : i32
  %shape = hlfir.shape_of %arg0 : (!hlfir.expr<?x?xi32>) -> !fir.shape<2>
  %mask = hlfir.elemental %shape : (!fir.shape<2>) -> !hlfir.expr<?x?x!fir.logical<4>> {
  ^bb0(%i: index, %j: index):
    %val = hlfir.apply %arg0, %i, %j : (!hlfir.expr<?x?xi32>, index, index) -> i32
    %cmp = arith.cmpi eq, %val, %c42 : i32
    %logical = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %logical : !fir.logical<4>
  }
  %res = hlfir.minloc %arg0 mask %mask : (!hlfir.expr<?x?xi32>, !hlfir.expr<?x?x!fir.logical<4>>) -> !hlfir.expr<2xi32>
  return %res : !hlfir.expr<2xi32>
}
// CHECK-LABEL: func.func @test_minloc_2d_equality_constant
// CHECK-DAG: %[[C42:.*]] = arith.constant 42 : i32
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[TRUE:.*]] = arith.constant true
// CHECK: %[[OUTER:.*]]:3 = fir.do_loop %[[IV1:.*]] = {{.*}} iter_args(%[[O1:.*]] = %[[C0]], %[[O2:.*]] = %[[C0]], %[[O3:.*]] = %[[TRUE]]) -> (i32, i32, i1)
// CHECK:   %[[INNER:.*]]:3 = fir.do_loop %[[IV2:.*]] = {{.*}} iter_args(%[[I1:.*]] = %[[O1]], %[[I2:.*]] = %[[O2]], %[[I3:.*]] = %[[O3]]) -> (i32, i32, i1)
// CHECK-NOT: hlfir.apply {{.*}} !fir.logical<4>
// CHECK:     %[[VAL:.*]] = hlfir.apply %{{.*}}, %[[IV2]], %[[IV1]]
// CHECK:     %[[EQ:.*]] = arith.cmpi eq, %[[VAL]], %[[C42]]
// CHECK:     %[[COND:.*]] = arith.andi %[[EQ]], %[[I3]]

// Rank 3: Constant: A == 42
func.func @test_minloc_3d_equality_constant(%arg0: !hlfir.expr<?x?x?xi32>) -> !hlfir.expr<3xi32> {
  %c42 = arith.constant 42 : i32
  %shape = hlfir.shape_of %arg0 : (!hlfir.expr<?x?x?xi32>) -> !fir.shape<3>
  %mask = hlfir.elemental %shape : (!fir.shape<3>) -> !hlfir.expr<?x?x?x!fir.logical<4>> {
  ^bb0(%i: index, %j: index, %k: index):
    %val = hlfir.apply %arg0, %i, %j, %k : (!hlfir.expr<?x?x?xi32>, index, index, index) -> i32
    %cmp = arith.cmpi eq, %val, %c42 : i32
    %logical = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %logical : !fir.logical<4>
  }
  %res = hlfir.minloc %arg0 mask %mask : (!hlfir.expr<?x?x?xi32>, !hlfir.expr<?x?x?x!fir.logical<4>>) -> !hlfir.expr<3xi32>
  return %res : !hlfir.expr<3xi32>
}
// CHECK-LABEL: func.func @test_minloc_3d_equality_constant
// CHECK-DAG: %[[C42:.*]] = arith.constant 42 : i32
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[TRUE:.*]] = arith.constant true
// CHECK: %[[OUT:.*]]:4 = fir.do_loop {{.*}} iter_args(%[[O1:.*]] = %[[C0]], %[[O2:.*]] = %[[C0]], %[[O3:.*]] = %[[C0]], %[[O4:.*]] = %[[TRUE]]) -> (i32, i32, i32, i1)
// CHECK:   %[[MID:.*]]:4 = fir.do_loop {{.*}} iter_args(%[[M1:.*]] = %[[O1]], %[[M2:.*]] = %[[O2]], %[[M3:.*]] = %[[O3]], %[[M4:.*]] = %[[O4]]) -> (i32, i32, i32, i1)
// CHECK:     %[[INN:.*]]:4 = fir.do_loop %[[IV3:.*]] = {{.*}} iter_args(%[[I1:.*]] = %[[M1]], %[[I2:.*]] = %[[M2]], %[[I3:.*]] = %[[M3]], %[[I4:.*]] = %[[M4]]) -> (i32, i32, i32, i1)
// CHECK-NOT: hlfir.apply {{.*}} !fir.logical<4>
// CHECK:       %[[VAL:.*]] = hlfir.apply %{{.*}}, %[[IV3]], %{{.*}}, %{{.*}}
// CHECK:       %[[EQ:.*]] = arith.cmpi eq, %[[VAL]], %[[C42]]
// CHECK:       %[[COND:.*]] = arith.andi %[[EQ]], %[[I4]]

// No Match: Result must be 0
func.func @test_minloc_no_match(%arg0: !hlfir.expr<?xi32>) -> !hlfir.expr<1xi32> {
  %c99 = arith.constant 99 : i32 
  %shape = hlfir.shape_of %arg0 : (!hlfir.expr<?xi32>) -> !fir.shape<1>
  %mask = hlfir.elemental %shape : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%i: index):
    %val = hlfir.apply %arg0, %i : (!hlfir.expr<?xi32>, index) -> i32
    %cmp = arith.cmpi eq, %val, %c99 : i32
    %logical = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %logical : !fir.logical<4>
  }
  %res = hlfir.minloc %arg0 mask %mask : (!hlfir.expr<?xi32>, !hlfir.expr<?x!fir.logical<4>>) -> !hlfir.expr<1xi32>
  return %res : !hlfir.expr<1xi32>
}
// CHECK-LABEL: func.func @test_minloc_no_match(
// CHECK-SAME: %[[ARRAY_NM:.*]]: !hlfir.expr<?xi32>)
// CHECK-DAG:    %[[C99:.*]] = arith.constant 99 : i32
// CHECK-DAG:    %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:    %[[TRUE:.*]] = arith.constant true
// CHECK: fir.do_loop %[[IV:.*]] = {{.*}} iter_args(%[[LOC:.*]] = %[[C0]], %[[FIRST:.*]] = %[[TRUE]]) -> (i32, i1)
// CHECK-NOT: hlfir.apply {{.*}} !fir.logical<4>
// CHECK:      %[[VAL:.*]] = hlfir.apply %[[ARRAY_NM]], %[[IV]]
// CHECK:      %[[MATCH:.*]] = arith.cmpi eq, %[[VAL]], %[[C99]] : i32
// CHECK:      %[[COND:.*]] = arith.andi %[[MATCH]], %[[FIRST]] : i1
// CHECK:      %[[IF_RES:.*]]:2 = fir.if %[[COND]] -> (i32, i1) {
// CHECK:        %[[CONV:.*]] = fir.convert %[[IV]]
// CHECK:        fir.result %[[CONV]], %false
// CHECK:      } else {
// CHECK:        fir.result %[[LOC]], %[[FIRST]] : i32, i1
// CHECK:      }

// First Match: Duplicate values
func.func @test_minloc_first_match(%arg0: !hlfir.expr<?xi32>, %target: i32) -> !hlfir.expr<1xi32> {
  %shape = hlfir.shape_of %arg0 : (!hlfir.expr<?xi32>) -> !fir.shape<1>
  %mask = hlfir.elemental %shape : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%i: index):
    %val = hlfir.apply %arg0, %i : (!hlfir.expr<?xi32>, index) -> i32
    %cmp = arith.cmpi eq, %val, %target : i32
    %logical = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %logical : !fir.logical<4>
  }
  %res = hlfir.minloc %arg0 mask %mask : (!hlfir.expr<?xi32>, !hlfir.expr<?x!fir.logical<4>>) -> !hlfir.expr<1xi32>
  return %res : !hlfir.expr<1xi32>
}
// CHECK-LABEL: func.func @test_minloc_first_match(
// CHECK-SAME: %[[ARRAY_FM:.*]]: !hlfir.expr<?xi32>, %[[TARGET_FM:.*]]: i32)
// CHECK-DAG:    %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:    %[[TRUE:.*]] = arith.constant true
// CHECK-DAG:    %[[FALSE:.*]] = arith.constant false
// Verify loop has only 2 iter_args (Coord, FirstHitFlag)
// CHECK: fir.do_loop %[[IV:.*]] = {{.*}} iter_args(%[[LOC:.*]] = %[[C0]], %[[FIRST:.*]] = %[[TRUE]]) -> (i32, i1)
// Verify mask elemental is bypassed
// CHECK-NOT: hlfir.apply {{.*}} !fir.logical<4>
// Verify the "Locking" logic: (Match AND is_first)
// CHECK:      %[[VAL:.*]] = hlfir.apply %[[ARRAY_FM]], %[[IV]]
// CHECK:      %[[MATCH:.*]] = arith.cmpi eq, %[[VAL]], %[[TARGET_FM]] : i32
// CHECK:      %[[COND:.*]] = arith.andi %[[MATCH]], %[[FIRST]] : i1
// Verify that once a match is found, we result in %false to lock it
// CHECK:      %[[IF_RES:.*]]:2 = fir.if %[[COND]] -> (i32, i1) {
// CHECK:        %[[CONV:.*]] = fir.convert %[[IV]]
// CHECK:        fir.result %[[CONV]], %[[FALSE]] : i32, i1
// CHECK:      } else {
// CHECK:        fir.result %[[LOC]], %[[FIRST]] : i32, i1
// CHECK:      }

// Negative test: Mask refers to a different array (%arg1) than the search 
// array (%arg0).
func.func @test_minloc_different_arrays(%arg0: !hlfir.expr<?xi32>, %arg1: !hlfir.expr<?xi32>, %target: i32) -> !hlfir.expr<1xi32> {
  %shape = hlfir.shape_of %arg0 : (!hlfir.expr<?xi32>) -> !fir.shape<1>
  %mask = hlfir.elemental %shape : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%i: index):
    // Optimization should fail here because %arg1 != %arg0
    %val_b = hlfir.apply %arg1, %i : (!hlfir.expr<?xi32>, index) -> i32
    %cmp = arith.cmpi eq, %val_b, %target : i32
    %logical = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %logical : !fir.logical<4>
  }
  %res = hlfir.minloc %arg0 mask %mask : (!hlfir.expr<?xi32>, !hlfir.expr<?x!fir.logical<4>>) -> !hlfir.expr<1xi32>
  return %res : !hlfir.expr<1xi32>
}
// CHECK-LABEL: func.func @test_minloc_different_arrays(
// CHECK-SAME: %[[ARRAY_A:.*]]: !hlfir.expr<?xi32>, %[[ARRAY_B:.*]]: !hlfir.expr<?xi32>, %[[TARGET:.*]]: i32)
// CHECK: %[[SENTINEL:.*]] = arith.constant 2147483647 : i32

// 1. Verify the loop uses three iter_args (Standard path: Loc, MinVal, FirstHit)
// CHECK: fir.do_loop %[[IV:.*]] = {{.*}} iter_args(%[[LOC:.*]] = {{.*}}, %[[MIN_VAL:.*]] = %[[SENTINEL]], %[[FIRST:.*]] = {{.*}}) -> (i32, i32, i1)

// 2. Verify the mask IS still applied (Optimization correctly skipped)
// CHECK: %[[MASK_VAL:.*]] = hlfir.apply {{.*}} : (!hlfir.expr<?x!fir.logical<4>>, index) -> !fir.logical<4>
// CHECK: %[[MASK_I1:.*]] = fir.convert %[[MASK_VAL]] : (!fir.logical<4>) -> i1

// 3. Verify the standard path's MINLOC comparison logic (slt instead of sgt)
// CHECK: fir.if %[[MASK_I1]] -> (i32, i32, i1) {
// CHECK:   %[[VAL_A:.*]] = hlfir.apply %[[ARRAY_A]], %[[IV]]
// CHECK:   %[[CMP:.*]] = arith.cmpi slt, %[[VAL_A]], %[[MIN_VAL]] : i32

// Negative Test: The target value is another array, so it is not invariant.
func.func @test_minloc_non_invariant_target(%arg0: !hlfir.expr<?xi32>, %arg1: !hlfir.expr<?xi32>) -> !hlfir.expr<1xi32> {
  %shape = hlfir.shape_of %arg0 : (!hlfir.expr<?xi32>) -> !fir.shape<1>
  %mask = hlfir.elemental %shape : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%i: index):
    %val_a = hlfir.apply %arg0, %i : (!hlfir.expr<?xi32>, index) -> i32
    %val_target = hlfir.apply %arg1, %i : (!hlfir.expr<?xi32>, index) -> i32
    // Optimization should fail here because %val_target is defined inside the 
    // elemental
    %cmp = arith.cmpi eq, %val_a, %val_target : i32
    %logical = fir.convert %cmp : (i1) -> !fir.logical<4>
    hlfir.yield_element %logical : !fir.logical<4>
  }
  %res = hlfir.minloc %arg0 mask %mask : (!hlfir.expr<?xi32>, !hlfir.expr<?x!fir.logical<4>>) -> !hlfir.expr<1xi32>
  return %res : !hlfir.expr<1xi32>
}
// CHECK-LABEL: func.func @test_minloc_non_invariant_target(
// CHECK-SAME: %[[ARRAY_A:.*]]: !hlfir.expr<?xi32>, %[[ARRAY_B:.*]]: !hlfir.expr<?xi32>)
// CHECK: %[[SENTINEL:.*]] = arith.constant 2147483647 : i32
// Verify the loop uses three iter_args (standard path)
// CHECK: fir.do_loop %[[IV:.*]] = {{.*}} iter_args(%[[LOC:.*]] = {{.*}}, %[[MAX:.*]] = %[[SENTINEL]], %[[FIRST:.*]] = {{.*}}) -> (i32, i32, i1)
// Verify the mask is still applied (because we couldn't inline the comparison)
// CHECK: %[[MASK_BIT:.*]] = hlfir.apply %{{.*}}, %[[IV]] : (!hlfir.expr<?x!fir.logical<4>>, index) -> !fir.logical<4>
// CHECK: %[[MASK_I1:.*]] = fir.convert %[[MASK_BIT]] : (!fir.logical<4>) -> i1
// CHECK: fir.if %[[MASK_I1]] -> (i32, i32, i1) {
// CHECK:   %[[VAL:.*]] = hlfir.apply %[[ARRAY_A]], %[[IV]]
// CHECK:   %[[CMP:.*]] = arith.cmpi slt, %[[VAL]], %[[MAX]]

