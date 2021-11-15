// RUN: fir-opt --split-input-file --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s | FileCheck %s
// RUN: fir-opt --split-input-file --fir-to-llvm-ir="target=aarch64-unknown-linux-gnu" %s | FileCheck %s
// RUN: fir-opt --split-input-file --fir-to-llvm-ir="target=i386-unknown-linux-gnu" %s | FileCheck %s
// RUN: fir-opt --split-input-file --fir-to-llvm-ir="target=powerpc64le-unknown-linux-gn" %s | FileCheck %s

//=============================================================================
// SUMMARY: Tests for FIR --> LLVM MLIR conversion independent of the target
//=============================================================================

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

// -----

// Test FIR complex addition conversion
// given: (x + iy) + (x' + iy')
// result: (x + x') + i(y + y')

func @fir_complex_add(%a: !fir.complex<16>, %b: !fir.complex<16>) -> !fir.complex<16> {
  %c = fir.addc %a, %b : !fir.complex<16>
  return %c : !fir.complex<16>
}

// CHECK-LABEL: llvm.func @fir_complex_add(
// CHECK-SAME:                             %[[ARG0:.*]]: !llvm.struct<(f128, f128)>, 
// CHECK-SAME:                             %[[ARG1:.*]]: !llvm.struct<(f128, f128)>) -> !llvm.struct<(f128, f128)> {
// CHECK:         %[[X0:.*]] = llvm.extractvalue %[[ARG0]][0 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         %[[Y0:.*]] = llvm.extractvalue %[[ARG0]][1 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         %[[X1:.*]] = llvm.extractvalue %[[ARG1]][0 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         %[[Y1:.*]] = llvm.extractvalue %[[ARG1]][1 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         %[[ADD_X0_X1:.*]] = llvm.fadd %[[X0]], %[[X1]]  : f128
// CHECK:         %[[ADD_Y0_Y1:.*]] = llvm.fadd %[[Y0]], %[[Y1]]  : f128
// CHECK:         %{{.*}} = llvm.mlir.undef : !llvm.struct<(f128, f128)>
// CHECK:         %{{.*}} = llvm.insertvalue %[[ADD_X0_X1]], %{{.*}}[0 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         %{{.*}} = llvm.insertvalue %[[ADD_Y0_Y1]], %{{.*}}[1 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         llvm.return %{{.*}} : !llvm.struct<(f128, f128)>

// -----

// Test FIR complex substraction conversion
// given: (x + iy) - (x' + iy')
// result: (x - x') + i(y - y')

func @fir_complex_sub(%a: !fir.complex<16>, %b: !fir.complex<16>) -> !fir.complex<16> {
  %c = fir.subc %a, %b : !fir.complex<16>
  return %c : !fir.complex<16>
}

// CHECK-LABEL: llvm.func @fir_complex_sub(
// CHECK-SAME:                             %[[ARG0:.*]]: !llvm.struct<(f128, f128)>, 
// CHECK-SAME:                             %[[ARG1:.*]]: !llvm.struct<(f128, f128)>) -> !llvm.struct<(f128, f128)> {
// CHECK:         %[[X0:.*]] = llvm.extractvalue %[[ARG0]][0 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         %[[Y0:.*]] = llvm.extractvalue %[[ARG0]][1 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         %[[X1:.*]] = llvm.extractvalue %[[ARG1]][0 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         %[[Y1:.*]] = llvm.extractvalue %[[ARG1]][1 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         %[[SUB_X0_X1:.*]] = llvm.fsub %[[X0]], %[[X1]]  : f128
// CHECK:         %[[SUB_Y0_Y1:.*]] = llvm.fsub %[[Y0]], %[[Y1]]  : f128
// CHECK:         %{{.*}} = llvm.mlir.undef : !llvm.struct<(f128, f128)>
// CHECK:         %{{.*}} = llvm.insertvalue %[[SUB_X0_X1]], %{{.*}}[0 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         %{{.*}} = llvm.insertvalue %[[SUB_Y0_Y1]], %{{.*}}[1 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         llvm.return %{{.*}} : !llvm.struct<(f128, f128)>

// -----

// Test FIR complex multiply conversion
// given: (x + iy) * (x' + iy')
// result: (xx'-yy')+i(xy'+yx')

func @fir_complex_mul(%a: !fir.complex<16>, %b: !fir.complex<16>) -> !fir.complex<16> {
  %c = fir.mulc %a, %b : !fir.complex<16>
  return %c : !fir.complex<16>
}

// CHECK-LABEL: llvm.func @fir_complex_mul(
// CHECK-SAME:                             %[[ARG0:.*]]: !llvm.struct<(f128, f128)>, 
// CHECK-SAME:                             %[[ARG1:.*]]: !llvm.struct<(f128, f128)>) -> !llvm.struct<(f128, f128)> {
// CHECK:         %[[X0:.*]] = llvm.extractvalue %[[ARG0]][0 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         %[[Y0:.*]] = llvm.extractvalue %[[ARG0]][1 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         %[[X1:.*]] = llvm.extractvalue %[[ARG1]][0 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         %[[Y1:.*]] = llvm.extractvalue %[[ARG1]][1 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         %[[MUL_X0_X1:.*]] = llvm.fmul %[[X0]], %[[X1]]  : f128
// CHECK:         %[[MUL_Y0_X1:.*]] = llvm.fmul %[[Y0]], %[[X1]]  : f128
// CHECK:         %[[MUL_X0_Y1:.*]] = llvm.fmul %[[X0]], %[[Y1]]  : f128
// CHECK:         %[[ADD:.*]] = llvm.fadd %[[MUL_X0_Y1]], %[[MUL_Y0_X1]]  : f128
// CHECK:         %[[MUL_Y0_Y1:.*]] = llvm.fmul %[[Y0]], %[[Y1]]  : f128
// CHECK:         %[[SUB:.*]] = llvm.fsub %[[MUL_X0_X1]], %[[MUL_Y0_Y1]]  : f128
// CHECK:         %{{.*}} = llvm.mlir.undef : !llvm.struct<(f128, f128)>
// CHECK:         %{{.*}} = llvm.insertvalue %[[SUB]], %{{.*}}[0 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         %{{.*}} = llvm.insertvalue %[[ADD]], %{{.*}}[1 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         llvm.return %{{.*}} : !llvm.struct<(f128, f128)>

// -----

// Test FIR complex division conversion
// given: (x + iy) / (x' + iy')
// result: ((xx'+yy')/d) + i((yx'-xy')/d) where d = x'x' + y'y'

func @fir_complex_div(%a: !fir.complex<16>, %b: !fir.complex<16>) -> !fir.complex<16> {
  %c = fir.divc %a, %b : !fir.complex<16>
  return %c : !fir.complex<16>
}

// CHECK-LABEL: llvm.func @fir_complex_div(
// CHECK-SAME:                             %[[ARG0:.*]]: !llvm.struct<(f128, f128)>, 
// CHECK-SAME:                             %[[ARG1:.*]]: !llvm.struct<(f128, f128)>) -> !llvm.struct<(f128, f128)> {
// CHECK:         %[[X0:.*]] = llvm.extractvalue %[[ARG0]][0 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         %[[Y0:.*]] = llvm.extractvalue %[[ARG0]][1 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         %[[X1:.*]] = llvm.extractvalue %[[ARG1]][0 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         %[[Y1:.*]] = llvm.extractvalue %[[ARG1]][1 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         %[[MUL_X0_X1:.*]] = llvm.fmul %[[X0]], %[[X1]]  : f128
// CHECK:         %[[MUL_X1_X1:.*]] = llvm.fmul %[[X1]], %[[X1]]  : f128
// CHECK:         %[[MUL_Y0_X1:.*]] = llvm.fmul %[[Y0]], %[[X1]]  : f128
// CHECK:         %[[MUL_X0_Y1:.*]] = llvm.fmul %[[X0]], %[[Y1]]  : f128
// CHECK:         %[[MUL_Y0_Y1:.*]] = llvm.fmul %[[Y0]], %[[Y1]]  : f128
// CHECK:         %[[MUL_Y1_Y1:.*]] = llvm.fmul %[[Y1]], %[[Y1]]  : f128
// CHECK:         %[[ADD_X1X1_Y1Y1:.*]] = llvm.fadd %[[MUL_X1_X1]], %[[MUL_Y1_Y1]]  : f128
// CHECK:         %[[ADD_X0X1_Y0Y1:.*]] = llvm.fadd %[[MUL_X0_X1]], %[[MUL_Y0_Y1]]  : f128
// CHECK:         %[[SUB_Y0X1_X0Y1:.*]] = llvm.fsub %[[MUL_Y0_X1]], %[[MUL_X0_Y1]]  : f128
// CHECK:         %[[DIV0:.*]] = llvm.fdiv %[[ADD_X0X1_Y0Y1]], %[[ADD_X1X1_Y1Y1]]  : f128
// CHECK:         %[[DIV1:.*]] = llvm.fdiv %[[SUB_Y0X1_X0Y1]], %[[ADD_X1X1_Y1Y1]]  : f128
// CHECK:         %{{.*}} = llvm.mlir.undef : !llvm.struct<(f128, f128)>
// CHECK:         %{{.*}} = llvm.insertvalue %[[DIV0]], %{{.*}}[0 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         %{{.*}} = llvm.insertvalue %[[DIV1]], %{{.*}}[1 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         llvm.return %{{.*}} : !llvm.struct<(f128, f128)>

// -----

// Test FIR complex negation conversion
// given: -(x + iy)
// result: -x - iy

func @fir_complex_neg(%a: !fir.complex<16>) -> !fir.complex<16> {
  %c = fir.negc %a : !fir.complex<16>
  return %c : !fir.complex<16>
}

// CHECK-LABEL: llvm.func @fir_complex_neg(
// CHECK-SAME:                             %[[ARG0:.*]]: !llvm.struct<(f128, f128)>) -> !llvm.struct<(f128, f128)> {
// CHECK:         %[[X:.*]] = llvm.extractvalue %[[ARG0]][0 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         %[[Y:.*]] = llvm.extractvalue %[[ARG0]][1 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         %[[NEGX:.*]] = llvm.fneg %[[X]]  : f128
// CHECK:         %[[NEGY:.*]] = llvm.fneg %[[Y]]  : f128
// CHECK:         %{{.*}} = llvm.insertvalue %[[NEGX]], %{{.*}}[0 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         %{{.*}} = llvm.insertvalue %[[NEGY]], %{{.*}}[1 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         llvm.return %{{.*}} : !llvm.struct<(f128, f128)>

// -----

// Test `fir.convert` operation conversion from Float type.

func @convert_from_float(%arg0 : f32) {
  %0 = fir.convert %arg0 : (f32) -> f16
  %1 = fir.convert %arg0 : (f32) -> f32
  %2 = fir.convert %arg0 : (f32) -> f64
  %3 = fir.convert %arg0 : (f32) -> f80
  %4 = fir.convert %arg0 : (f32) -> f128
  %5 = fir.convert %arg0 : (f32) -> i1
  %6 = fir.convert %arg0 : (f32) -> i8
  %7 = fir.convert %arg0 : (f32) -> i16
  %8 = fir.convert %arg0 : (f32) -> i32
  %9 = fir.convert %arg0 : (f32) -> i64
  return
}

// CHECK-LABEL: convert_from_float(
// CHECK-SAME:                     %[[ARG0:.*]]: f32
// CHECK:         %{{.*}} = llvm.fptrunc %[[ARG0]] : f32 to f16
// CHECK-NOT:     f32 to f32
// CHECK:         %{{.*}} = llvm.fpext %[[ARG0]] : f32 to f64
// CHECK:         %{{.*}} = llvm.fpext %[[ARG0]] : f32 to f80
// CHECK:         %{{.*}} = llvm.fpext %[[ARG0]] : f32 to f128
// CHECK:         %{{.*}} = llvm.fptosi %[[ARG0]] : f32 to i1
// CHECK:         %{{.*}} = llvm.fptosi %[[ARG0]] : f32 to i8
// CHECK:         %{{.*}} = llvm.fptosi %[[ARG0]] : f32 to i16
// CHECK:         %{{.*}} = llvm.fptosi %[[ARG0]] : f32 to i32
// CHECK:         %{{.*}} = llvm.fptosi %[[ARG0]] : f32 to i64

// -----

// Test `fir.convert` operation conversion from Integer type.

func @convert_from_int(%arg0 : i32) {
  %0 = fir.convert %arg0 : (i32) -> f16
  %1 = fir.convert %arg0 : (i32) -> f32
  %2 = fir.convert %arg0 : (i32) -> f64
  %3 = fir.convert %arg0 : (i32) -> f80
  %4 = fir.convert %arg0 : (i32) -> f128
  %5 = fir.convert %arg0 : (i32) -> i1
  %6 = fir.convert %arg0 : (i32) -> i8
  %7 = fir.convert %arg0 : (i32) -> i16
  %8 = fir.convert %arg0 : (i32) -> i32
  %9 = fir.convert %arg0 : (i32) -> i64
  %10 = fir.convert %arg0 : (i32) -> i64
  %ptr = fir.convert %10 : (i64) -> !fir.ref<i64>
  return
}

// CHECK-LABEL: convert_from_int(
// CHECK-SAME:                   %[[ARG0:.*]]: i32
// CHECK:         %{{.*}} = llvm.sitofp %[[ARG0]] : i32 to f16
// CHECK:         %{{.*}} = llvm.sitofp %[[ARG0]] : i32 to f32
// CHECK:         %{{.*}} = llvm.sitofp %[[ARG0]] : i32 to f64
// CHECK:         %{{.*}} = llvm.sitofp %[[ARG0]] : i32 to f80
// CHECK:         %{{.*}} = llvm.sitofp %[[ARG0]] : i32 to f128
// CHECK:         %{{.*}} = llvm.trunc %[[ARG0]] : i32 to i1
// CHECK:         %{{.*}} = llvm.trunc %[[ARG0]] : i32 to i8
// CHECK:         %{{.*}} = llvm.trunc %[[ARG0]] : i32 to i16
// CHECK-NOT:     %{{.*}} = llvm.trunc %[[ARG0]] : i32 to i32
// CHECK:         %{{.*}} = llvm.sext %[[ARG0]] : i32 to i64
// CHECK:         %{{.*}} = llvm.inttoptr %{{.*}} : i64 to !llvm.ptr<i64>

// -----

// Test `fir.convert` operation conversion from !fir.ref<> type.

func @convert_from_ref(%arg0 : !fir.ref<i32>) {
  %0 = fir.convert %arg0 : (!fir.ref<i32>) -> !fir.ref<i8>
  %1 = fir.convert %arg0 : (!fir.ref<i32>) -> i32
  return
}

// CHECK-LABEL: convert_from_ref(
// CHECK-SAME:                   %[[ARG0:.*]]: !llvm.ptr<i32>
// CHECK:         %{{.*}} = llvm.bitcast %[[ARG0]] : !llvm.ptr<i32> to !llvm.ptr<i8>
// CHECK:         %{{.*}} = llvm.ptrtoint %[[ARG0]] : !llvm.ptr<i32> to i32

// -----

// Test `fir.convert` operation conversion between fir.complex types.

func @convert_complex4(%arg0 : !fir.complex<4>) -> !fir.complex<8> {
  %0 = fir.convert %arg0 : (!fir.complex<4>) -> !fir.complex<8>
  return %0 : !fir.complex<8>
}

// CHECK-LABEL: func @convert_complex4(
// CHECK-SAME:                         %[[ARG0:.*]]: !llvm.struct<(f32, f32)>) -> !llvm.struct<(f64, f64)>
// CHECK:         %[[X:.*]] = llvm.extractvalue %[[ARG0]][0 : i32] : !llvm.struct<(f32, f32)>
// CHECK:         %[[Y:.*]] = llvm.extractvalue %[[ARG0]][1 : i32] : !llvm.struct<(f32, f32)>
// CHECK:         %[[CONVERTX:.*]] = llvm.fpext %[[X]] : f32 to f64
// CHECK:         %[[CONVERTY:.*]] = llvm.fpext %[[Y]] : f32 to f64
// CHECK:         %[[STRUCT0:.*]] = llvm.mlir.undef : !llvm.struct<(f64, f64)>
// CHECK:         %[[STRUCT1:.*]] = llvm.insertvalue %[[CONVERTX]], %[[STRUCT0]][0 : i32] : !llvm.struct<(f64, f64)>
// CHECK:         %[[STRUCT2:.*]] = llvm.insertvalue %[[CONVERTY]], %[[STRUCT1]][1 : i32] : !llvm.struct<(f64, f64)>
// CHECK:         llvm.return %[[STRUCT2]] : !llvm.struct<(f64, f64)>

// Test `fir.convert` operation conversion between fir.complex types.

func @convert_complex16(%arg0 : !fir.complex<16>) -> !fir.complex<2> {
  %0 = fir.convert %arg0 : (!fir.complex<16>) -> !fir.complex<2>
  return %0 : !fir.complex<2>
}

// CHECK-LABEL: func @convert_complex16(
// CHECK-SAME:                          %[[ARG0:.*]]: !llvm.struct<(f128, f128)>) -> !llvm.struct<(f16, f16)>
// CHECK:         %[[X:.*]] = llvm.extractvalue %[[ARG0]][0 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         %[[Y:.*]] = llvm.extractvalue %[[ARG0]][1 : i32] : !llvm.struct<(f128, f128)>
// CHECK:         %[[CONVERTX:.*]] = llvm.fptrunc %[[X]] : f128 to f16
// CHECK:         %[[CONVERTY:.*]] = llvm.fptrunc %[[Y]] : f128 to f16
// CHECK:         %[[STRUCT0:.*]] = llvm.mlir.undef : !llvm.struct<(f16, f16)>
// CHECK:         %[[STRUCT1:.*]] = llvm.insertvalue %[[CONVERTX]], %[[STRUCT0]][0 : i32] : !llvm.struct<(f16, f16)>
// CHECK:         %[[STRUCT2:.*]] = llvm.insertvalue %[[CONVERTY]], %[[STRUCT1]][1 : i32] : !llvm.struct<(f16, f16)>
// CHECK:         llvm.return %[[STRUCT2]] : !llvm.struct<(f16, f16)>

// -----

// Test `fir.store` --> `llvm.store` conversion

func @test_store_index(%val_to_store : index, %addr : !fir.ref<index>) {
  fir.store %val_to_store to %addr : !fir.ref<index>
  return
}

// CHECK-LABEL:   llvm.func @test_store_index
// CHECK-SAME:    (%[[arg0:.*]]: i64, %[[arg1:.*]]: !llvm.ptr<i64>) {
// CHECK-NEXT:    llvm.store %[[arg0]], %[[arg1]] : !llvm.ptr<i64>
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }

func @test_store_box(%array : !fir.ref<!fir.box<!fir.array<?x?xf32>>>, %box : !fir.box<!fir.array<?x?xf32>>) {
  fir.store %box to %array : !fir.ref<!fir.box<!fir.array<?x?xf32>>>
  return
}

// CHECK-LABEL:  llvm.func @test_store_box
// CHECK-SAME:  (%[[arg0:.*]]: !llvm.ptr<struct<(ptr<f{{.*}}>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<2 x array<3 x i{{.*}}>>)>>,
// CHECK-SAME:  %[[arg1:.*]]: !llvm.ptr<struct<(ptr<f{{.*}}>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<2 x array<3 x i{{.*}}>>)>>) {
// CHECK-NEXT:  %[[box_to_store:.*]] = llvm.load %arg1 : !llvm.ptr<struct<(ptr<f{{.*}}>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<2 x array<3 x i{{.*}}>>)>>
// CHECK-NEXT:  llvm.store %[[box_to_store]], %[[arg0]] : !llvm.ptr<struct<(ptr<f{{.*}}>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<2 x array<3 x i{{.*}}>>)>>
// CHECK-NEXT:  llvm.return
// CHECK-NEXT:  }

// -----

// Test `fir.load` --> `llvm.load` conversion

func @test_load_index(%addr : !fir.ref<index>) {
  %0 = fir.load %addr : !fir.ref<index>
  return
}

// CHECK-LABEL: llvm.func @test_load_index(
// CHECK-SAME:  %[[arg1:.*]]: !llvm.ptr<i64>) {
// CHECK-NEXT:    %0 = llvm.load %[[arg1]] : !llvm.ptr<i64>
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }

func @test_load_box(%addr : !fir.ref<!fir.box<!fir.array<10xf32>>>) {
  %0 = fir.load %addr : !fir.ref<!fir.box<!fir.array<10xf32>>>
  return
}

// Loading a `fir.ref<!fir.box>> is a no-op
// CHECK-LABEL: llvm.func @test_load_box
// CHECK-SAME: (%{{.*}}: !llvm.ptr<struct<(ptr<array<10 x f{{.*}}>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i{{.*}}>>)>>) {
// CHECK-NEXT:  llvm.return
// CHECK-NEXT:  }

// -----

// Test `fir.box_rank` conversion.

func @extract_rank(%arg0: !fir.box<!fir.array<*:f64>>) -> i32 {
  %0 = fir.box_rank %arg0 : (!fir.box<!fir.array<*:f64>>) -> i32
  return %0 : i32
}

// CHECK-LABEL: llvm.func @extract_rank(
// CHECK-SAME:                          %[[ARG0:.*]]: !llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>) -> i32
// CHECK:         %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[CRANK:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:         %[[GEP:.*]] = llvm.getelementptr %[[ARG0]][%[[C0]], %[[CRANK]]] : (!llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>, i32, i32) -> !llvm.ptr<i32>
// CHECK:         %[[RANK:.*]] = llvm.load %[[GEP]] : !llvm.ptr<i32>
// CHECK:         llvm.return %[[RANK]] : i32

// -----

// Test `fir.box_addr` conversion.

func @extract_addr(%arg0: !fir.box<!fir.array<*:f64>>) -> !fir.ref<f64> {
  %0 = fir.box_addr %arg0 : (!fir.box<!fir.array<*:f64>>) -> !fir.ref<f64>
  return %0 : !fir.ref<f64>
}

// CHECK-LABEL: llvm.func @extract_addr(
// CHECK-SAME:                          %[[ARG0:.*]]: !llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>) -> !llvm.ptr<f64>
// CHECK:         %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[CADDR:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[GEP:.*]] = llvm.getelementptr %[[ARG0]][%[[C0]], %[[CADDR]]] : (!llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>, i32, i32) -> !llvm.ptr<ptr<f64>>
// CHECK:         %[[ADDR:.*]] = llvm.load %[[GEP]] : !llvm.ptr<ptr<f64>>
// CHECK:         llvm.return %[[ADDR]] : !llvm.ptr<f64>

// -----

// Test `fir.box_dims` conversion.

func @extract_dims(%arg0: !fir.box<!fir.array<*:f64>>) -> index {
  %c1 = arith.constant 0 : i32
  %0:3 = fir.box_dims %arg0, %c1 : (!fir.box<!fir.array<*:f64>>, i32) -> (index, index, index)
  return %0 : index
}

// CHECK-LABEL: llvm.func @extract_dims(
// CHECK-SAME:                          %[[ARG0:.*]]: !llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>) -> i64
// CHECK:         %[[C0_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[CDIMS:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK:         %[[C0_2:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[GEP0:.*]] = llvm.getelementptr %[[ARG0]][%[[C0]], %[[CDIMS]], %[[C0_1]], %[[C0_2]]] : (!llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>, i32, i32, i32, i32) -> !llvm.ptr<i64>
// CHECK:         %[[LOAD0:.*]] = llvm.load %[[GEP0]] : !llvm.ptr<i64>
// CHECK:         %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:         %[[GEP1:.*]] = llvm.getelementptr %[[ARG0]][%[[C0]], %[[CDIMS]], %[[C0_1]], %[[C1]]] : (!llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>, i32, i32, i32, i32) -> !llvm.ptr<i64>
// CHECK:         %[[LOAD1:.*]] = llvm.load %[[GEP1]] : !llvm.ptr<i64>
// CHECK:         %[[C2:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:         %[[GEP2:.*]] = llvm.getelementptr %[[ARG0]][%[[C0]], %[[CDIMS]], %[[C0_1]], %[[C2]]] : (!llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>, i32, i32, i32, i32) -> !llvm.ptr<i64>
// CHECK:         %[[LOAD2:.*]] = llvm.load %[[GEP2]] : !llvm.ptr<i64>
// CHECK:         llvm.return %[[LOAD0]] : i64

// -----

// Test `fir.box_elesize` conversion.

func @extract_elesize(%arg0: !fir.box<f32>) -> i32 {
  %0 = fir.box_elesize %arg0 : (!fir.box<f32>) -> i32
  return %0 : i32
}

// CHECK-LABEL: llvm.func @extract_elesize(
// CHECK-SAME:                             %[[ARG0:.*]]: !llvm.ptr<struct<(ptr<f32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>) -> i32
// CHECK:         %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[CELESIZE:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:         %[[GEP:.*]] = llvm.getelementptr %[[ARG0]][%[[C0]], %[[CELESIZE]]] : (!llvm.ptr<struct<(ptr<f32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>, i32, i32) -> !llvm.ptr<i32>
// CHECK:         %[[ELE_SIZE:.*]] = llvm.load %[[GEP]] : !llvm.ptr<i32>
// CHECK:         llvm.return %[[ELE_SIZE]] : i32

// -----

// Test `fir.box_isarray` conversion.
// `rank` is extracted from `fir.box` and compare to 0.

func @box_isarray(%arg0: !fir.box<!fir.array<*:f64>>) -> i1 {
  %0 = fir.box_isarray %arg0 : (!fir.box<!fir.array<*:f64>>) -> i1
  return %0 : i1
}

// CHECK-LABEL: llvm.func @box_isarray(
// CHECK-SAME:                         %[[ARG0:.*]]: !llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>) -> i1
// CHECK:         %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[RANKPOS:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:         %[[GEP:.*]] = llvm.getelementptr %[[ARG0]][%[[C0]], %[[RANKPOS]]] : (!llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>, i32, i32) -> !llvm.ptr<i32> 
// CHECK:         %[[RANK:.*]] = llvm.load %[[GEP]] : !llvm.ptr<i32>
// CHECK:         %[[C0_ISARRAY:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[IS_ARRAY:.*]] = llvm.icmp "ne" %[[RANK]], %[[C0_ISARRAY]] : i32
// CHECK:         llvm.return %[[IS_ARRAY]] : i1

// -----

// Test `fir.box_isalloc` conversion.
// `attribute` is extracted from `fir.box` and checked against a mask equal to
// the value of `CFI_attribute_allocatable`.

func @box_isalloc(%arg0: !fir.box<!fir.array<*:f64>>) -> i1 {
  %0 = fir.box_isalloc %arg0 : (!fir.box<!fir.array<*:f64>>) -> i1
  return %0 : i1
}

// CHECK-LABEL: llvm.func @box_isalloc(
// CHECK-SAME:                         %[[ARG0:.*]]: !llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>) -> i1
// CHECK:         %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[ATTRPOS:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK:         %[[GEP:.*]] = llvm.getelementptr %[[ARG0]][%[[C0]], %[[ATTRPOS]]] : (!llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>, i32, i32) -> !llvm.ptr<i32>
// CHECK:         %[[ATTR:.*]] = llvm.load %[[GEP]] : !llvm.ptr<i32>
// CHECK:         %[[ATTR_ISALLOC:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:         %[[AND:.*]] = llvm.and %[[ATTR]], %[[ATTR_ISALLOC]]  : i32
// CHECK:         %[[CMP_C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[IS_ALLOC:.*]] = llvm.icmp "ne" %[[AND]], %[[CMP_C0]] : i32
// CHECK:         llvm.return %[[IS_ALLOC]] : i1

// -----

// Test `fir.box_isptr` conversion.
// `attribute` is extracted from `fir.box` and checked against a mask equal to
// the value of `CFI_attribute_pointer`.

func @box_isptr(%arg0: !fir.box<!fir.array<*:f64>>) -> i1 {
  %0 = fir.box_isptr %arg0 : (!fir.box<!fir.array<*:f64>>) -> i1
  return %0 : i1
}

// CHECK-LABEL: llvm.func @box_isptr(
// CHECK-SAME:                         %[[ARG0:.*]]: !llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>) -> i1
// CHECK:         %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[ATTRPOS:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK:         %[[GEP:.*]] = llvm.getelementptr %[[ARG0]][%[[C0]], %[[ATTRPOS]]] : (!llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>, i32, i32) -> !llvm.ptr<i32>
// CHECK:         %[[ATTR:.*]] = llvm.load %[[GEP]] : !llvm.ptr<i32>
// CHECK:         %[[ATTR_ISALLOC:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:         %[[AND:.*]] = llvm.and %[[ATTR]], %[[ATTR_ISALLOC]]  : i32
// CHECK:         %[[CMP_C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[IS_ALLOC:.*]] = llvm.icmp "ne" %[[AND]], %[[CMP_C0]] : i32
// CHECK:         llvm.return %[[IS_ALLOC]] : i1

// -----

// Test fir.alloca of one element

func @alloca_one() -> !fir.ref<i32> {
  %1 = fir.alloca i32
  return %1 : !fir.ref<i32>
}

// CHECK-LABEL: llvm.func @alloca_one() -> !llvm.ptr<i32>
// CHECK: [[N:%.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: [[A:%.*]] = llvm.alloca [[N]] x i32
// CHECK: llvm.return [[A]] : !llvm.ptr<i32>

// -----

// Test fir.alloca of several elements

func @alloca_several() -> !fir.ref<i32> {
  %0 = arith.constant 100 : index
  %1 = fir.alloca i32, %0
  return %1 : !fir.ref<i32>
}

// CHECK-LABEL: llvm.func @alloca_several() -> !llvm.ptr<i32>
// CHECK: [[N:%.*]] = llvm.mlir.constant(100 : index) : i64
// CHECK: [[ONE:%.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: [[TOTAL:%.*]] = llvm.mul [[ONE]], [[N]] : i64
// CHECK: [[A:%.*]] = llvm.alloca [[TOTAL]] x i32
// CHECK: llvm.return [[A]] : !llvm.ptr<i32>

// -----

// Test fir.alloca of pointer to array

func @alloca_ptr_to_array() -> !fir.ref<!fir.ptr<!fir.array<?xi32>>> {
  %1 = fir.alloca !fir.ptr<!fir.array<?xi32>>
  return %1 : !fir.ref<!fir.ptr<!fir.array<?xi32>>>
}

// CHECK-LABEL: llvm.func @alloca_ptr_to_array() -> !llvm.ptr<ptr<i32>>
// CHECK: [[ONE:%.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: [[A:%.*]] = llvm.alloca [[ONE]] x !llvm.ptr<i32>
// CHECK: llvm.return [[A]] : !llvm.ptr<ptr<i32>>

// -----

// Test fir.alloca of char array

func @alloca_char_array(%l: i32, %e : index) -> !fir.ref<!fir.array<?x?x!fir.char<1,?>>> {
  %a = fir.alloca !fir.array<?x?x!fir.char<1,?>>(%l : i32), %e, %e
  return %a :  !fir.ref<!fir.array<?x?x!fir.char<1,?>>>
}

// CHECK-LABEL: llvm.func @alloca_char_array
// CHECK-SAME: ([[L:%.*]]: i32, [[E:%.*]]: i64) -> !llvm.ptr<i8>
// CHECK-DAG: [[UNUSEDONE:%.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG: [[LCAST:%.*]] = llvm.sext [[L]] : i32 to i64
// CHECK-DAG: [[ONE:%.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: [[PROD1:%.*]] = llvm.mul [[LCAST]], [[ONE]] : i64
// CHECK: [[PROD2:%.*]] = llvm.mul [[PROD1]], [[E]] : i64
// CHECK: [[PROD3:%.*]] = llvm.mul [[PROD2]], [[E]] : i64
// CHECK: [[A:%.*]] = llvm.alloca [[PROD3]] x i8 {in_type = !fir.array<?x?x!fir.char<1,?>>
// CHECK: return [[A]] : !llvm.ptr<i8>

// -----

// Test fir.alloca of record type with LEN parameters
//   type t(p1,p2)
//      integer, len :: p1
//      integer(kind=2), len :: p2
//      integer f1
//      real f2
//   end type t

func private @_QTtP.mem.size(%0 : i32, %1 : i16) -> index

func @alloca_record(%arg0 : i32, %arg1 : i16) -> !fir.ref<!fir.type<_QTt(p1:i32,p2:i16){f1:i32,f2:f32}>> {
  %0 = fir.alloca !fir.type<_QTt(p1:i32,p2:i16){f1:i32,f2:f32}>(%arg0, %arg1 : i32, i16) {name = "_QEvar"}
  return %0 : !fir.ref<!fir.type<_QTt(p1:i32,p2:i16){f1:i32,f2:f32}>>
}

// CHECK-LABEL: llvm.func @alloca_record
// CHECK-SAME: ([[ARG0:%.*]]: i32, [[ARG1:%.*]]: i16)
// CHECK-SAME: -> !llvm.ptr<struct<"_QTt", (i32, f32)>>
// CHECK: [[SIZE:%.*]] = llvm.call @_QTtP.mem.size([[ARG0]], [[ARG1]]) : (i32, i16) -> i64
// CHECK: [[ALLOC:%.*]] = llvm.alloca [[SIZE]] x i8
// CHECK: [[A:%.*]] = llvm.bitcast [[ALLOC]] : !llvm.ptr<i8> to !llvm.ptr<struct<"_QTt", (i32, f32)>>
// CHECK: llvm.return [[A]] : !llvm.ptr<struct<"_QTt", (i32, f32)>>

// -----

// Test fir.alloca of a multidimensional array, with operands

func @alloca_multidim_array(%0 : index) -> !fir.ref<!fir.array<8x16x32xf32>> {
  %1 = arith.constant 24 : index
  %2 = fir.alloca !fir.array<8x16x32xf32>, %0, %1
  return %2 : !fir.ref<!fir.array<8x16x32xf32>>
}

// CHECK-LABEL: llvm.func @alloca_multidim_array
// CHECK-SAME: ([[OP1:%.*]]: i64) -> !llvm.ptr<array<32 x array<16 x array<8 x f32>
// CHECK: [[OP2:%.*]] = llvm.mlir.constant(24 : index) : i64
// CHECK: [[ONE:%.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: [[ALL:%.*]] = llvm.mlir.constant(4096 : i64) : i64
// CHECK: [[MUL1:%.*]] = llvm.mul [[ONE]], [[ALL]] : i64
// CHECK: [[MUL2:%.*]] = llvm.mul [[MUL1]], [[OP1]] : i64
// CHECK: [[TOTAL:%.*]] = llvm.mul [[MUL2]], [[OP2]] : i64
// CHECK: [[A:%.*]] = llvm.alloca [[TOTAL]] x !llvm.array<32 x array<16 x array<8 x f32>
// CHECK: llvm.return [[A]] : !llvm.ptr<array<32 x array<16 x array<8 x f32>

// -----

// Test alloca with an array with holes.
// Constant factor of 60 (4*3*5) must be included.

func @alloca_array_with_holes(%0 : index, %1 : index) -> !fir.ref<!fir.array<4x?x3x?x5xi32>> {
  %a = fir.alloca !fir.array<4x?x3x?x5xi32>, %0, %1
  return %a : !fir.ref<!fir.array<4x?x3x?x5xi32>>
}

// CHECK-LABEL: llvm.func @alloca_array_with_holes
// CHECK-SAME: ([[A:%.*]]: i64, [[B:%.*]]: i64) -> !llvm.ptr<i32>
// CHECK-DAG: [[ONE:%.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG: [[FIXED:%.*]] = llvm.mlir.constant(60 : i64) : i64
// CHECK: [[PROD1:%.*]] = llvm.mul [[ONE]], [[FIXED]] : i64
// CHECK: [[PROD2:%.*]] = llvm.mul [[PROD1]], [[A]] : i64
// CHECK: [[PROD3:%.*]] = llvm.mul [[PROD2]], [[B]] : i64
// CHECK: [[RES:%.*]] = llvm.alloca [[PROD3]] x i32 {in_type = !fir.array<4x?x3x?x5xi32>
// CHECK: llvm.return [[RES]] : !llvm.ptr<i32>

// -----

// Test `fir.select_case` operation conversion with INTEGER.

func @select_case_integer(%arg0: !fir.ref<i32>) -> i32 {
  %2 = fir.load %arg0 : !fir.ref<i32>
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c4_i32 = arith.constant 4 : i32
  %c5_i32 = arith.constant 5 : i32
  %c7_i32 = arith.constant 7 : i32
  %c8_i32 = arith.constant 8 : i32
  %c15_i32 = arith.constant 15 : i32
  %c21_i32 = arith.constant 21 : i32
  fir.select_case %2 : i32 [#fir.upper, %c1_i32, ^bb1,
                            #fir.point, %c2_i32, ^bb2,
                            #fir.interval, %c4_i32, %c5_i32, ^bb4,
                            #fir.point, %c7_i32, ^bb5,
                            #fir.interval, %c8_i32, %c15_i32, ^bb5,
                            #fir.lower, %c21_i32, ^bb5,
                            unit, ^bb3]
^bb1:  // pred: ^bb0
  %c1_i32_0 = arith.constant 1 : i32
  fir.store %c1_i32_0 to %arg0 : !fir.ref<i32>
  br ^bb6
^bb2:  // pred: ^bb0
  %c2_i32_1 = arith.constant 2 : i32
  fir.store %c2_i32_1 to %arg0 : !fir.ref<i32>
  br ^bb6
^bb3:  // pred: ^bb0
  %c0_i32 = arith.constant 0 : i32
  fir.store %c0_i32 to %arg0 : !fir.ref<i32>
  br ^bb6
^bb4:  // pred: ^bb0
  %c4_i32_2 = arith.constant 4 : i32
  fir.store %c4_i32_2 to %arg0 : !fir.ref<i32>
  br ^bb6
^bb5:  // 3 preds: ^bb0, ^bb0, ^bb0
  %c7_i32_3 = arith.constant 7 : i32
  fir.store %c7_i32_3 to %arg0 : !fir.ref<i32>
  br ^bb6
^bb6:  // 5 preds: ^bb1, ^bb2, ^bb3, ^bb4, ^bb5
  %3 = fir.load %arg0 : !fir.ref<i32>
  return %3 : i32
}

// CHECK-LABEL: llvm.func @select_case_integer(
// CHECK-SAME:                         %[[ARG0:.*]]: !llvm.ptr<i32>) -> i32 {
// CHECK:         %[[SELECT_VALUE:.*]] = llvm.load %[[ARG0]] : !llvm.ptr<i32>
// CHECK:         %[[CST1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:         %[[CST2:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:         %[[CST4:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK:         %[[CST5:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK:         %[[CST7:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK:         %[[CST8:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK:         %[[CST15:.*]] = llvm.mlir.constant(15 : i32) : i32
// CHECK:         %[[CST21:.*]] = llvm.mlir.constant(21 : i32) : i32
// Check for upper bound `case (:1)`
// CHECK:         %[[CMP_SLE:.*]] = llvm.icmp "sle" %[[SELECT_VALUE]], %[[CST1]] : i32
// CHECK:         llvm.cond_br %[[CMP_SLE]], ^bb2, ^bb1
// CHECK-LABEL: ^bb1:
// Check for point value `case (2)`
// CHECK:         %[[CMP_EQ:.*]] = llvm.icmp "eq" %[[SELECT_VALUE]], %[[CST2]] : i32
// CHECK:         llvm.cond_br %[[CMP_EQ]], ^bb4, ^bb3
// Block ^bb1 in original FIR code.
// CHECK-LABEL: ^bb2:
// CHECK:         llvm.br ^bb{{.*}}
// CHECK-LABEL: ^bb3:
// Check for the lower bound for the interval `case (4:5)`
// CHECK:        %[[CMP_SLE:.*]] = llvm.icmp "sle" %[[CST4]], %[[SELECT_VALUE]] : i32
// CHECK:        llvm.cond_br %[[CMP_SLE]], ^bb[[UPPERBOUND5:.*]], ^bb7
// Block ^bb2 in original FIR code.
// CHECK-LABEL: ^bb4:
// CHECK:        llvm.br ^bb{{.*}}
// Block ^bb3 in original FIR code.
// CHECK-LABEL: ^bb5:
// CHECK:        llvm.br ^bb{{.*}}
// CHECK:       ^bb[[UPPERBOUND5]]:
// Check for the upper bound for the interval `case (4:5)`
// CHECK:        %[[CMP_SLE:.*]] = llvm.icmp "sle" %[[SELECT_VALUE]], %[[CST5]] : i32
// CHECK:        llvm.cond_br %[[CMP_SLE]], ^bb8, ^bb7
// CHECK-LABEL: ^bb7:
// Check for the point value 7 in `case (7,8:15,21:)`
// CHECK: %[[CMP_EQ:.*]] = llvm.icmp "eq" %[[SELECT_VALUE]], %[[CST7]] : i32
// CHECK: llvm.cond_br %[[CMP_EQ]], ^bb13, ^bb9
// Block ^bb4 in original FIR code.
// CHECK-LABEL: ^bb8:
// CHECK:        llvm.br ^bb{{.*}}
// CHECK-LABEL: ^bb9:
// Check for lower bound 8 in `case (7,8:15,21:)`
// CHECK:        %[[CMP_SLE:.*]] = llvm.icmp "sle" %[[CST8]], %[[SELECT_VALUE]] : i32
// CHECK:        llvm.cond_br %[[CMP_SLE]], ^bb[[INTERVAL8_15:.*]], ^bb11
// CHECK:       ^bb[[INTERVAL8_15]]:
// Check for upper bound 15 in `case (7,8:15,21:)`
// CHECK:        %[[CMP_SLE:.*]] = llvm.icmp "sle" %[[SELECT_VALUE]], %[[CST15]] : i32
// CHECK:        llvm.cond_br %[[CMP_SLE]], ^bb13, ^bb11
// CHECK-LABEL: ^bb11:
// Check for lower bound 21 in `case (7,8:15,21:)`
// CHECK:        %[[CMP_SLE:.*]]  = llvm.icmp "sle" %[[CST21]], %[[SELECT_VALUE]] : i32
// CHECK:        llvm.cond_br %[[CMP_SLE]], ^bb13, ^bb12
// CHECK-LABEL: ^bb12:
// CHECK:         llvm.br ^bb5
// Block ^bb5 in original FIR code.
// CHECK-LABEL: ^bb13:
// CHECK:        llvm.br ^bb14
// Block ^bb6 in original FIR code.
// CHECK-LABEL: ^bb14:
// CHECK:        %[[RET:.*]] = llvm.load %[[ARG0:.*]] : !llvm.ptr<i32>
// CHECK:        llvm.return %[[RET]] : i32

// -----

// Test `fir.select_case` operation conversion with LOGICAL.

func @select_case_logical(%arg0: !fir.ref<!fir.logical<4>>) {
  %1 = fir.load %arg0 : !fir.ref<!fir.logical<4>>
  %2 = fir.convert %1 : (!fir.logical<4>) -> i1
  %false = arith.constant false
  %true = arith.constant true
  fir.select_case %2 : i1 [#fir.point, %false, ^bb1,
                           #fir.point, %true, ^bb2,
                            unit, ^bb3]
^bb1:
  %c1_i32 = arith.constant 1 : i32
  br ^bb3
^bb2:
  %c2_i32 = arith.constant 2 : i32
  br ^bb3
^bb3:
  return
}

// CHECK-LABEL: llvm.func @select_case_logical(
// CHECK-SAME:                                 %[[ARG0:.*]]: !llvm.ptr<i32>
// CHECK:         %[[LOAD_ARG0:.*]] = llvm.load %[[ARG0]] : !llvm.ptr<i32>
// CHECK:         %[[SELECT_VALUE:.*]] = llvm.trunc %[[LOAD_ARG0]] : i32 to i1
// CHECK:         %[[CST_FALSE:.*]] = llvm.mlir.constant(false) : i1
// CHECK:         %[[CST_TRUE:.*]] = llvm.mlir.constant(true) : i1
// CHECK:         %[[CMPEQ:.*]] = llvm.icmp "eq" %[[SELECT_VALUE]], %[[CST_FALSE]] : i1
// CHECK:         llvm.cond_br %[[CMPEQ]], ^bb2, ^bb1
// CHECK-LABEL: ^bb1:
// CHECK:         %[[CMPEQ:.*]] = llvm.icmp "eq" %[[SELECT_VALUE]], %[[CST_TRUE]] : i1
// CHECK:         llvm.cond_br %[[CMPEQ]], ^bb4, ^bb3
// CHECK-LABEL: ^bb2:
// CHECK:         llvm.br ^bb5
// CHECK-LABEL: ^bb3:
// CHECK:         llvm.br ^bb5
// CHECK-LABEL: ^bb4:
// CHECK:         llvm.br ^bb5
// CHECK-LABEL: ^bb5:
// CHECK:        llvm.return

// -----

// Test `fir.is_present`

func @test_is_present_i64(%arg0: !fir.ref<i64>) -> () {
  %0 = fir.is_present %arg0 : (!fir.ref<i64>) -> i1
  return
}

// CHECK-LABEL: @test_is_present_i64
// CHECK-SAME: (%[[arg:.*]]: !llvm.ptr<i64>)
// CHECK-NEXT:  %[[constant:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:  %[[ptr:.*]] = llvm.ptrtoint %[[arg]] : !llvm.ptr<i64> to i64
// CHECK-NEXT:  %{{.*}} = llvm.icmp "ne" %[[ptr]], %[[constant]] : i64
// CHECK-NEXT:  llvm.return
// CHECK-NEXT: }

func @test_is_present_box(%arg0: !fir.box<!fir.ref<i64>>) -> () {
  %0 = fir.is_present %arg0 : (!fir.box<!fir.ref<i64>>) -> i1
  return
}

// CHECK-LABEL: @test_is_present_box
// CHECK-SAME: (%[[arg:.*]]: !llvm.ptr<struct<(ptr<i64>, i64, i32, i8, i8, i8, i8)>>)
// CHECK-NEXT: %[[constant:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[ptr:.*]] = llvm.ptrtoint %[[arg]] : !llvm.ptr<struct<(ptr<i64>, i64, i32, i8, i8, i8, i8)>> to i64
// CHECK-NEXT: %{{.*}} = llvm.icmp "ne" %[[ptr]], %[[constant]] : i64
// CHECK-NEXT: llvm.return
// CHECK-NEXT: }


// -----

// Test `fir.absent`

func @test_absent_i64() -> () {
  %0 = fir.absent !fir.ref<i64>
  return
}

// CHECK-LABEL: @test_absent_i64
// CHECK-NEXT:  %{{.*}} = llvm.mlir.null : !llvm.ptr<i64>
// CHECK-NEXT:  llvm.return
// CHECK-NEXT:  }

func @test_absent_box() -> () {
  %0 = fir.absent !fir.box<!fir.array<?xf32>>
  return
}
// CHECK-LABEL: @test_absent_box
// CHECK-NEXT:  %{{.*}} = llvm.mlir.null : !llvm.ptr<struct<(ptr<f32>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>>
// CHECK-NEXT:  llvm.return
// CHECK-NEXT:  }

// -----

// This is a bit more comprehensive test for `fir.is_present` and `fir.absent`
// when used together

func @is_present(%arg0: !fir.ref<i64>) -> i1 {
  %0 = fir.is_present %arg0 : (!fir.ref<i64>) -> i1
  return %0 : i1
}

// CHECK-LABEL: @is_present
// CHECK-SAME: (%[[arg:.*]]: !llvm.ptr<i64>) -> i1
// CHECK-NEXT:  %[[constant:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:  %[[ptr:.*]] = llvm.ptrtoint %[[arg]] : !llvm.ptr<i64> to i64
// CHECK-NEXT:  %[[ret_val:.*]] = llvm.icmp "ne" %[[ptr]], %[[constant]] : i64
// CHECK-NEXT:  llvm.return %[[ret_val]] : i1
// CHECK-NEXT: }

func @absent() -> i1 {
  %0 = fir.absent !fir.ref<i64>
  %1 = fir.call @is_present(%0) : (!fir.ref<i64>) -> i1
  return %1 : i1
}

// CHECK-LABEL: @absent
// CHECK-SAME:  () -> i1
// CHECK-NEXT:  %[[ptr:.*]] = llvm.mlir.null : !llvm.ptr<i64>
// CHECK-NEXT:  %[[ret_val:.*]] = llvm.call @is_present(%[[ptr]]) : (!llvm.ptr<i64>) -> i1
// CHECK-NEXT:  llvm.return %[[ret_val]] : i1
