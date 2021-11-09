// RUN: fir-opt --split-input-file --fir-to-llvm-ir %s | FileCheck %s

// Test simple global LLVM conversion

fir.global @g_i0 : i32 {
  %1 = arith.constant 0 : i32
  fir.has_value %1 : i32
}

// CHECK: llvm.mlir.global external @g_i0() : i32 {
// CHECK:   %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:   llvm.return %[[C0]] : i32
// CHECK: }

// -----

fir.global @g_ci5 constant : i32 {
  %c = arith.constant 5 : i32
  fir.has_value %c : i32
}

// CHECK: llvm.mlir.global external constant @g_ci5() : i32 {
// CHECK:   %[[C5:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK:   llvm.return %[[C5]] : i32
// CHECK: }

// -----

fir.global internal @i_i515 (515:i32) : i32
// CHECK: llvm.mlir.global internal @i_i515(515 : i32) : i32

// -----

fir.global common @C_i511 (0:i32) : i32
// CHECK: llvm.mlir.global common @C_i511(0 : i32) : i32

// -----

fir.global weak @w_i86 (86:i32) : i32
// CHECK: llvm.mlir.global weak @w_i86(86 : i32) : i32

// -----

fir.global linkonce @w_i86 (86:i32) : i32
// CHECK: llvm.mlir.global linkonce @w_i86(86 : i32) : i32

// -----

// Test conversion of fir.address_of with fir.global

func @f1() {
  %0 = fir.address_of(@symbol) : !fir.ref<i64>
  return
}

fir.global @symbol : i64 {
  %0 = arith.constant 1 : i64
  fir.has_value %0 : i64
}

// CHECK: %{{.*}} = llvm.mlir.addressof @[[SYMBOL:.*]] : !llvm.ptr<i64>

// CHECK: llvm.mlir.global external @[[SYMBOL]]() : i64 {
// CHECK:   %{{.*}} = llvm.mlir.constant(1 : i64) : i64
// CHECK:   llvm.return %{{.*}} : i64
// CHECK: }

// -----

// Test global with insert_on_range operation covering the full array
// in initializer region.

fir.global internal @_QEmultiarray : !fir.array<32x32xi32> {
  %c0_i32 = arith.constant 1 : i32
  %0 = fir.undefined !fir.array<32x32xi32>
  %2 = fir.insert_on_range %0, %c0_i32, [0 : index, 31 : index, 0 : index, 31 : index] : (!fir.array<32x32xi32>, i32) -> !fir.array<32x32xi32>
  fir.has_value %2 : !fir.array<32x32xi32>
}

// CHECK: llvm.mlir.global internal @_QEmultiarray() : !llvm.array<32 x array<32 x i32>> {
// CHECK:   %[[CST:.*]] = llvm.mlir.constant(dense<1> : vector<32x32xi32>) : !llvm.array<32 x array<32 x i32>>
// CHECK:   llvm.return %[[CST]] : !llvm.array<32 x array<32 x i32>>
// CHECK: }

// -----

// Test global with insert_on_range operation not covering the full array
// in initializer region.

fir.global internal @_QEmultiarray : !fir.array<32xi32> {
  %c0_i32 = arith.constant 1 : i32
  %0 = fir.undefined !fir.array<32xi32>
  %2 = fir.insert_on_range %0, %c0_i32, [5 : index, 31 : index] : (!fir.array<32xi32>, i32) -> !fir.array<32xi32>
  fir.has_value %2 : !fir.array<32xi32>
}

// CHECK:          llvm.mlir.global internal @_QEmultiarray() : !llvm.array<32 x i32> {
// CHECK:            %[[CST:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:            %{{.*}} = llvm.mlir.undef : !llvm.array<32 x i32>
// CHECK:            %{{.*}} = llvm.insertvalue %[[CST]], %{{.*}}[5] : !llvm.array<32 x i32>
// CHECK-COUNT-24:   %{{.*}} = llvm.insertvalue %[[CST]], %{{.*}}[{{.*}}] : !llvm.array<32 x i32>
// CHECK:            %{{.*}} = llvm.insertvalue %[[CST]], %{{.*}}[31] : !llvm.array<32 x i32>
// CHECK-NOT:        llvm.insertvalue
// CHECK:            llvm.return %{{.*}} : !llvm.array<32 x i32>
// CHECK:          }

// -----

// Test fir.zero_bits operation with LLVM ptr type

func @zero_test_ptr() {
  %z = fir.zero_bits !llvm.ptr<f32>
  return
}

// CHECK: %{{.*}} = llvm.mlir.null : !llvm.ptr<f32>
// CHECK-NOT: fir.zero_bits

// -----

// Test fir.zero_bits operation with integer type.

func @zero_test_integer() {
  %z0 = fir.zero_bits i8
  %z1 = fir.zero_bits i16
  %z2 = fir.zero_bits i32
  %z3 = fir.zero_bits i64
  return
}

// CHECK: %{{.*}} = llvm.mlir.constant(0 : i8) : i8
// CHECK: %{{.*}} = llvm.mlir.constant(0 : i16) : i16
// CHECK: %{{.*}} = llvm.mlir.constant(0 : i32) : i32
// CHECK: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
// CHECK-NOT: fir.zero_bits

// -----

// Test fir.zero_bits operation with floating points types.

func @zero_test_float() {
  %z0 = fir.zero_bits f16
  %z1 = fir.zero_bits bf16
  %z2 = fir.zero_bits f32
  %z3 = fir.zero_bits f64
  %z4 = fir.zero_bits f80
  %z5 = fir.zero_bits f128
  return
}

// CHECK: %{{.*}} = llvm.mlir.constant(0.000000e+00 : f16) : f16
// CHECK: %{{.*}} = llvm.mlir.constant(0.000000e+00 : bf16) : bf16
// CHECK: %{{.*}} = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK: %{{.*}} = llvm.mlir.constant(0.000000e+00 : f64) : f64
// CHECK: %{{.*}} = llvm.mlir.constant(0.000000e+00 : f80) : f80
// CHECK: %{{.*}} = llvm.mlir.constant(0.000000e+00 : f128) : f128
// CHECK-NOT: fir.zero_bits

// -----

// Verify that fir.unreachable is transformed to llvm.unreachable

// CHECK:  llvm.func @test_unreachable() {
// CHECK-NEXT:    llvm.unreachable
// CHECK-NEXT:  }

func @test_unreachable() {
  fir.unreachable
}

// -----

// Test `fir.select` operation conversion pattern.
// Check that the if-then-else ladder is correctly constructed and that we
// branch to the correct block.

func @select(%arg : index, %arg2 : i32) -> i32 {
  %0 = arith.constant 1 : i32
  %1 = arith.constant 2 : i32
  %2 = arith.constant 3 : i32
  %3 = arith.constant 4 : i32
  fir.select %arg:index [ 1, ^bb1(%0:i32),
                          2, ^bb2(%2,%arg,%arg2:i32,index,i32),
                          3, ^bb3(%arg2,%2:i32,i32),
                          4, ^bb4(%1:i32),
                          unit, ^bb5 ]
  ^bb1(%a : i32) :
    return %a : i32
  ^bb2(%b : i32, %b2 : index, %b3:i32) :
    %castidx = arith.index_cast %b2 : index to i32
    %4 = arith.addi %b, %castidx : i32
    %5 = arith.addi %4, %b3 : i32
    return %5 : i32
  ^bb3(%c:i32, %c2:i32) :
    %6 = arith.addi %c, %c2 : i32
    return %6 : i32
  ^bb4(%d : i32) :
    return %d : i32
  ^bb5 :
    %zero = arith.constant 0 : i32
    return %zero : i32
}

// CHECK-LABEL: func @select(
// CHECK-SAME:               %[[SELECTVALUE:.*]]: [[IDX:.*]],
// CHECK-SAME:               %[[ARG1:.*]]: i32)
// CHECK:         %[[C0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:         %[[C1:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:         %[[C2:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:         %[[SELECTOR:.*]] = llvm.trunc %[[SELECTVALUE]] : i{{.*}} to i32
// CHECK:         llvm.switch %[[SELECTOR]], ^bb5 [
// CHECK:           1: ^bb1(%[[C0]] : i32),
// CHECK:           2: ^bb2(%[[C2]], %[[SELECTVALUE]], %[[ARG1]] : i32, [[IDX]], i32),
// CHECK:           3: ^bb3(%[[ARG1]], %[[C2]] : i32, i32),
// CHECK:           4: ^bb4(%[[C1]] : i32)
// CHECK:         ]

// -----

// Test `fir.select_rank` operation conversion pattern.
// Check that the if-then-else ladder is correctly constructed and that we
// branch to the correct block.

func @select_rank(%arg : i32, %arg2 : i32) -> i32 {
  %0 = arith.constant 1 : i32
  %1 = arith.constant 2 : i32
  %2 = arith.constant 3 : i32
  %3 = arith.constant 4 : i32
  fir.select_rank %arg:i32 [ 1, ^bb1(%0:i32),
                             2, ^bb2(%2,%arg,%arg2:i32,i32,i32),
                             3, ^bb3(%arg2,%2:i32,i32),
                             4, ^bb4(%1:i32),
                             unit, ^bb5 ]
  ^bb1(%a : i32) :
    return %a : i32
  ^bb2(%b : i32, %b2 : i32, %b3:i32) :
    %4 = arith.addi %b, %b2 : i32
    %5 = arith.addi %4, %b3 : i32
    return %5 : i32
  ^bb3(%c:i32, %c2:i32) :
    %6 = arith.addi %c, %c2 : i32
    return %6 : i32
  ^bb4(%d : i32) :
    return %d : i32
  ^bb5 :
    %zero = arith.constant 0 : i32
    return %zero : i32
}

// CHECK-LABEL: func @select_rank(
// CHECK-SAME:                    %[[SELECTVALUE:.*]]: i32,
// CHECK-SAME:                    %[[ARG1:.*]]: i32)
// CHECK:         %[[C0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:         %[[C1:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:         %[[C2:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:         llvm.switch %[[SELECTVALUE]], ^bb5 [
// CHECK:           1: ^bb1(%[[C0]] : i32),
// CHECK:           2: ^bb2(%[[C2]], %[[SELECTVALUE]], %[[ARG1]] : i32, i32, i32),
// CHECK:           3: ^bb3(%[[ARG1]], %[[C2]] : i32, i32),
// CHECK:           4: ^bb4(%[[C1]] : i32)
// CHECK:         ]

// -----

// Test fir.extract_value operation conversion with derived type.

func @extract_derived_type() -> f32 {
  %0 = fir.undefined !fir.type<derived{f:f32}>
  %1 = fir.extract_value %0, ["f", !fir.type<derived{f:f32}>] : (!fir.type<derived{f:f32}>) -> f32
  return %1 : f32
}

// CHECK-LABEL: llvm.func @extract_derived_type
// CHECK:         %[[STRUCT:.*]] = llvm.mlir.undef : !llvm.struct<"derived", (f32)>
// CHECK:         %[[VALUE:.*]] = llvm.extractvalue %[[STRUCT]][0 : i32] : !llvm.struct<"derived", (f32)>
// CHECK:         llvm.return %[[VALUE]] : f32

// -----

// Test fir.extract_value operation conversion with a multi-dimensional array
// of tuple.

func @extract_array(%a : !fir.array<10x10xtuple<i32, f32>>) -> f32 {
  %0 = fir.extract_value %a, [5 : index, 4 : index, 1 : index] : (!fir.array<10x10xtuple<i32, f32>>) -> f32
  return %0 : f32
}

// CHECK-LABEL: llvm.func @extract_array(
// CHECK-SAME:                           %[[ARR:.*]]: !llvm.array<10 x array<10 x struct<(i32, f32)>>>
// CHECK:         %[[VALUE:.*]] = llvm.extractvalue %[[ARR]][4 : index, 5 : index, 1 : index] : !llvm.array<10 x array<10 x struct<(i32, f32)>>>
// CHECK:         llvm.return %[[VALUE]] : f32

// -----

// Test fir.insert_value operation conversion with a multi-dimensional array
// of tuple.

func @extract_array(%a : !fir.array<10x10xtuple<i32, f32>>) {
  %f = arith.constant 2.0 : f32
  %i = arith.constant 1 : i32
  %0 = fir.insert_value %a, %i, [5 : index, 4 : index, 0 : index] : (!fir.array<10x10xtuple<i32, f32>>, i32) -> !fir.array<10x10xtuple<i32, f32>>
  %1 = fir.insert_value %a, %f, [5 : index, 4 : index, 1 : index] : (!fir.array<10x10xtuple<i32, f32>>, f32) -> !fir.array<10x10xtuple<i32, f32>>
  return
}

// CHECK-LABEL: llvm.func @extract_array(
// CHECK-SAME:                           %[[ARR:.*]]: !llvm.array<10 x array<10 x struct<(i32, f32)>>>
// CHECK:         %{{.*}} = llvm.insertvalue %{{.*}}, %[[ARR]][4 : index, 5 : index, 0 : index] : !llvm.array<10 x array<10 x struct<(i32, f32)>>>
// CHECK:         %{{.*}} = llvm.insertvalue %{{.*}}, %[[ARR]][4 : index, 5 : index, 1 : index] : !llvm.array<10 x array<10 x struct<(i32, f32)>>>
// CHECK:         llvm.return

// -----

// Test fir.insert_value operation conversion with derived type.

func @insert_tuple(%a : tuple<i32, f32>) {
  %f = arith.constant 2.0 : f32
  %1 = fir.insert_value %a, %f, [1 : index] : (tuple<i32, f32>, f32) -> tuple<i32, f32>
  return
}

// CHECK-LABEL: func @insert_tuple(
// CHECK-SAME:                     %[[TUPLE:.*]]: !llvm.struct<(i32, f32)>
// CHECK:         %{{.*}} = llvm.insertvalue %{{.*}}, %[[TUPLE]][1 : index] : !llvm.struct<(i32, f32)>
// CHECK:         llvm.return

// -----
// Test `fir.call` -> `llvm.call` conversion for functions that take no arguments
// and return nothing

func @dummy_basic() {
  return
}

func @test_call_basic() {
  fir.call @dummy_basic() : () -> ()
  return
}

// CHECK-LABEL: func @test_call_basic() {
// CHECK-NEXT:  llvm.call @dummy_basic() : () -> ()
// CHECK-NEXT:  return
// CHECK-NEXT: }

// Test `fir.call` -> `llvm.call` conversion for functions that take one
// argument and return nothing

func @dummy_with_arg(%arg0 : i32) {
  return
}

func @test_call_with_arg(%arg0 : i32) {
  fir.call @dummy_with_arg(%arg0) : (i32) -> ()
  return
}

// CHECK-LABEL:  llvm.func @test_call_with_arg(%arg0: i32) {
// CHECK-NEXT:    llvm.call @dummy_with_arg(%arg0) : (i32) -> ()
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }

// Test `fir.call` -> `llvm.call` conversion for functions that take no
// arguments, but return a value

func @dummy_return_val() -> i32 {
  %1 = arith.constant 123 : i32
  return %1 : i32
}

func @test_call_return_val() -> i32 {
  %1 = fir.call @dummy_return_val() : () -> (i32)
  return %1 : i32
}

// CHECK-LABEL:  llvm.func @test_call_return_val() -> i32 {
// CHECK-NEXT:    %0 = llvm.call @dummy_return_val() : () -> i32
// CHECK-NEXT:    llvm.return %0 : i32
// CHECK-NEXT:  }
