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

func.func @f1() {
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
  %2 = fir.insert_on_range %0, %c0_i32 from (0, 0) to (31, 31) : (!fir.array<32x32xi32>, i32) -> !fir.array<32x32xi32>
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
  %2 = fir.insert_on_range %0, %c0_i32 from (5) to (31) : (!fir.array<32xi32>, i32) -> !fir.array<32xi32>
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

func.func @zero_test_ptr() {
  %z = fir.zero_bits !llvm.ptr<f32>
  return
}

// CHECK: %{{.*}} = llvm.mlir.null : !llvm.ptr<f32>
// CHECK-NOT: fir.zero_bits

// -----

// Test fir.zero_bits operation with integer type.

func.func @zero_test_integer() {
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

func.func @zero_test_float() {
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

// Verify that fir.allocmem is transformed to a call to malloc
// and that fir.freemem is transformed to a call to free
// Single item case

func.func @test_alloc_and_freemem_one() {
  %z0 = fir.allocmem i32
  fir.freemem %z0 : !fir.heap<i32>
  return
}

// CHECK-LABEL:  llvm.func @test_alloc_and_freemem_one() {
// CHECK-NEXT:    [[N:%.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    llvm.call @malloc([[N]])
// CHECK:         llvm.call @free(%{{.*}})
// CHECK-NEXT:    llvm.return

// -----
// Verify that fir.allocmem is transformed to a call to malloc
// and that fir.freemem is transformed to a call to free
// Several item case

func.func @test_alloc_and_freemem_several() {
  %z0 = fir.allocmem !fir.array<100xf32>
  fir.freemem %z0 : !fir.heap<!fir.array<100xf32>>
  return
}

// CHECK-LABEL:  llvm.func @test_alloc_and_freemem_several() {
// CHECK: [[NULL:%.*]]  = llvm.mlir.null : !llvm.ptr<array<100 x f32>>
// CHECK: [[PTR:%.*]]  = llvm.getelementptr [[NULL]][{{.*}}] : (!llvm.ptr<array<100 x f32>>, i64) -> !llvm.ptr<array<100 x f32>>
// CHECK: [[N:%.*]]  = llvm.ptrtoint [[PTR]] : !llvm.ptr<array<100 x f32>> to i64
// CHECK: [[MALLOC:%.*]] = llvm.call @malloc([[N]])
// CHECK: [[B1:%.*]] = llvm.bitcast [[MALLOC]] : !llvm.ptr<i8> to !llvm.ptr<array<100 x f32>>
// CHECK: [[B2:%.*]] = llvm.bitcast [[B1]] : !llvm.ptr<array<100 x f32>> to !llvm.ptr<i8>
// CHECK:              llvm.call @free([[B2]])
// CHECK:              llvm.return


func.func @test_with_shape(%ncols: index, %nrows: index) {
  %1 = fir.allocmem !fir.array<?x?xf32>, %ncols, %nrows
  fir.freemem %1 : !fir.heap<!fir.array<?x?xf32>>
  return
}

// CHECK-LABEL: llvm.func @test_with_shape
// CHECK-SAME: %[[NCOLS:.*]]: i64, %[[NROWS:.*]]: i64
// CHECK:   %[[FOUR:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK:   %[[DIM1_SIZE:.*]] = llvm.mul %[[FOUR]], %[[NCOLS]]  : i64
// CHECK:   %[[TOTAL_SIZE:.*]] = llvm.mul %[[DIM1_SIZE]], %[[NROWS]]  : i64
// CHECK:   %[[MEM:.*]] = llvm.call @malloc(%[[TOTAL_SIZE]])
// CHECK:   %[[B1:.*]] = llvm.bitcast %[[MEM]] : !llvm.ptr<i8> to !llvm.ptr<f32>
// CHECK:   %[[B2:.*]] = llvm.bitcast %[[B1]] : !llvm.ptr<f32> to !llvm.ptr<i8>
// CHECK:   llvm.call @free(%[[B2]]) : (!llvm.ptr<i8>) -> ()
// CHECK:   llvm.return
// CHECK: }

func.func @test_string_with_shape(%len: index, %nelems: index) {
  %1 = fir.allocmem !fir.array<?x!fir.char<1,?>>(%len : index), %nelems
  fir.freemem %1 : !fir.heap<!fir.array<?x!fir.char<1,?>>>
  return
}

// CHECK-LABEL: llvm.func @test_string_with_shape
// CHECK-SAME: %[[LEN:.*]]: i64, %[[NELEMS:.*]]: i64)
// CHECK:   %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:   %[[LEN_SIZE:.*]] = llvm.mul %[[ONE]], %[[LEN]]  : i64
// CHECK:   %[[TOTAL_SIZE:.*]] = llvm.mul %[[LEN_SIZE]], %[[NELEMS]]  : i64
// CHECK:   %[[MEM:.*]] = llvm.call @malloc(%[[TOTAL_SIZE]])
// CHECK:   %[[B1:.*]] = llvm.bitcast %[[MEM]] : !llvm.ptr<i8> to !llvm.ptr<i8>
// CHECK:   %[[B2:.*]] = llvm.bitcast %[[B1]] : !llvm.ptr<i8> to !llvm.ptr<i8>
// CHECK:   llvm.call @free(%[[B2]]) : (!llvm.ptr<i8>) -> ()
// CHECK:   llvm.return
// CHECK: }

// -----

// Verify that fir.unreachable is transformed to llvm.unreachable

func.func @test_unreachable() {
  fir.unreachable
}

// CHECK:  llvm.func @test_unreachable() {
// CHECK-NEXT:    llvm.unreachable
// CHECK-NEXT:  }

// -----

// Test `fir.select` operation conversion pattern.
// Check that the if-then-else ladder is correctly constructed and that we
// branch to the correct block.

func.func @select(%arg : index, %arg2 : i32) -> i32 {
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
// CHECK:         llvm.switch %[[SELECTOR]] : i32, ^bb5 [
// CHECK:           1: ^bb1(%[[C0]] : i32),
// CHECK:           2: ^bb2(%[[C2]], %[[SELECTVALUE]], %[[ARG1]] : i32, [[IDX]], i32),
// CHECK:           3: ^bb3(%[[ARG1]], %[[C2]] : i32, i32),
// CHECK:           4: ^bb4(%[[C1]] : i32)
// CHECK:         ]

// -----

// Test `fir.select_rank` operation conversion pattern.
// Check that the if-then-else ladder is correctly constructed and that we
// branch to the correct block.

func.func @select_rank(%arg : i32, %arg2 : i32) -> i32 {
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
// CHECK:         llvm.switch %[[SELECTVALUE]] : i32, ^bb5 [
// CHECK:           1: ^bb1(%[[C0]] : i32),
// CHECK:           2: ^bb2(%[[C2]], %[[SELECTVALUE]], %[[ARG1]] : i32, i32, i32),
// CHECK:           3: ^bb3(%[[ARG1]], %[[C2]] : i32, i32),
// CHECK:           4: ^bb4(%[[C1]] : i32)
// CHECK:         ]

// -----

// Test fir.extract_value operation conversion with derived type.

func.func @extract_derived_type() -> f32 {
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

func.func @extract_array(%a : !fir.array<10x10xtuple<i32, f32>>) -> f32 {
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

func.func @extract_array(%a : !fir.array<10x10xtuple<i32, f32>>) {
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

func.func @insert_tuple(%a : tuple<i32, f32>) {
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

func.func @dummy_basic() {
  return
}

func.func @test_call_basic() {
  fir.call @dummy_basic() : () -> ()
  return
}

// CHECK-LABEL: func @test_call_basic() {
// CHECK-NEXT:  llvm.call @dummy_basic() : () -> ()
// CHECK-NEXT:  return
// CHECK-NEXT: }

// Test `fir.call` -> `llvm.call` conversion for functions that take one
// argument and return nothing

func.func @dummy_with_arg(%arg0 : i32) {
  return
}

func.func @test_call_with_arg(%arg0 : i32) {
  fir.call @dummy_with_arg(%arg0) : (i32) -> ()
  return
}

// CHECK-LABEL:  llvm.func @test_call_with_arg(%arg0: i32) {
// CHECK-NEXT:    llvm.call @dummy_with_arg(%arg0) : (i32) -> ()
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }

// Test `fir.call` -> `llvm.call` conversion for functions that take no
// arguments, but return a value

func.func @dummy_return_val() -> i32 {
  %1 = arith.constant 123 : i32
  return %1 : i32
}

func.func @test_call_return_val() -> i32 {
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

func.func @fir_complex_add(%a: !fir.complex<16>, %b: !fir.complex<16>) -> !fir.complex<16> {
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

func.func @fir_complex_sub(%a: !fir.complex<16>, %b: !fir.complex<16>) -> !fir.complex<16> {
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

func.func @fir_complex_mul(%a: !fir.complex<16>, %b: !fir.complex<16>) -> !fir.complex<16> {
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

func.func @fir_complex_div(%a: !fir.complex<16>, %b: !fir.complex<16>) -> !fir.complex<16> {
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

func.func @fir_complex_neg(%a: !fir.complex<16>) -> !fir.complex<16> {
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

// Test FIR complex compare conversion

func.func @compare_complex_eq(%a : !fir.complex<8>, %b : !fir.complex<8>) -> i1 {
  %r = fir.cmpc "oeq", %a, %b : !fir.complex<8>
  return %r : i1
}

// CHECK-LABEL: llvm.func @compare_complex_eq
// CHECK-SAME: [[A:%.*]]: !llvm.struct<(f64, f64)>,
// CHECK-SAME: [[B:%.*]]: !llvm.struct<(f64, f64)>
// CHECK-DAG: [[RA:%.*]] = llvm.extractvalue [[A]][0 : i32] : !llvm.struct<(f64, f64)>
// CHECK-DAG: [[IA:%.*]] = llvm.extractvalue [[A]][1 : i32] : !llvm.struct<(f64, f64)>
// CHECK-DAG: [[RB:%.*]] = llvm.extractvalue [[B]][0 : i32] : !llvm.struct<(f64, f64)>
// CHECK-DAG: [[IB:%.*]] = llvm.extractvalue [[B]][1 : i32] : !llvm.struct<(f64, f64)>
// CHECK-DAG: [[RESR:%.*]] = llvm.fcmp "oeq" [[RA]], [[RB]] : f64
// CHECK-DAG: [[RESI:%.*]] = llvm.fcmp "oeq" [[IA]], [[IB]] : f64
// CHECK: [[RES:%.*]] = llvm.and [[RESR]], [[RESI]] : i1
// CHECK: return [[RES]] : i1

func.func @compare_complex_ne(%a : !fir.complex<8>, %b : !fir.complex<8>) -> i1 {
  %r = fir.cmpc "une", %a, %b : !fir.complex<8>
  return %r : i1
}

// CHECK-LABEL: llvm.func @compare_complex_ne
// CHECK-SAME: [[A:%.*]]: !llvm.struct<(f64, f64)>,
// CHECK-SAME: [[B:%.*]]: !llvm.struct<(f64, f64)>
// CHECK-DAG: [[RA:%.*]] = llvm.extractvalue [[A]][0 : i32] : !llvm.struct<(f64, f64)>
// CHECK-DAG: [[IA:%.*]] = llvm.extractvalue [[A]][1 : i32] : !llvm.struct<(f64, f64)>
// CHECK-DAG: [[RB:%.*]] = llvm.extractvalue [[B]][0 : i32] : !llvm.struct<(f64, f64)>
// CHECK-DAG: [[IB:%.*]] = llvm.extractvalue [[B]][1 : i32] : !llvm.struct<(f64, f64)>
// CHECK-DAG: [[RESR:%.*]] = llvm.fcmp "une" [[RA]], [[RB]] : f64
// CHECK-DAG: [[RESI:%.*]] = llvm.fcmp "une" [[IA]], [[IB]] : f64
// CHECK: [[RES:%.*]] = llvm.or [[RESR]], [[RESI]] : i1
// CHECK: return [[RES]] : i1

func.func @compare_complex_other(%a : !fir.complex<8>, %b : !fir.complex<8>) -> i1 {
  %r = fir.cmpc "ogt", %a, %b : !fir.complex<8>
  return %r : i1
}

// CHECK-LABEL: llvm.func @compare_complex_other
// CHECK-SAME: [[A:%.*]]: !llvm.struct<(f64, f64)>,
// CHECK-SAME: [[B:%.*]]: !llvm.struct<(f64, f64)>
// CHECK-DAG: [[RA:%.*]] = llvm.extractvalue [[A]][0 : i32] : !llvm.struct<(f64, f64)>
// CHECK-DAG: [[RB:%.*]] = llvm.extractvalue [[B]][0 : i32] : !llvm.struct<(f64, f64)>
// CHECK: [[RESR:%.*]] = llvm.fcmp "ogt" [[RA]], [[RB]] : f64
// CHECK: return [[RESR]] : i1

// -----

// Test `fir.convert` operation conversion from Float type.

func.func @convert_from_float(%arg0 : f32) {
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

func.func @convert_from_int(%arg0 : i32) {
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

func.func @convert_from_ref(%arg0 : !fir.ref<i32>) {
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

func.func @convert_complex4(%arg0 : !fir.complex<4>) -> !fir.complex<8> {
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

func.func @convert_complex16(%arg0 : !fir.complex<16>) -> !fir.complex<2> {
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

// Test constc.

func.func @test_constc4() -> !fir.complex<4> {
  %0 = fir.constc (#fir.real<4, 1.4>, #fir.real<4, 2.3>) : !fir.complex<4>
  return %0 : !fir.complex<4>
}

// CHECK-LABEL: @test_constc4
// CHECK_SAME: () -> !llvm.struct<(f32, f32)>
// CHECK-DAG: [[rp:%.*]] = llvm.mlir.constant(1.400000e+00 : f32) : f32
// CHECK-DAG: [[ip:%.*]] = llvm.mlir.constant(2.300000e+00 : f32) : f32
// CHECK: [[undef:%.*]] = llvm.mlir.undef : !llvm.struct<(f32, f32)>
// CHECK: [[withr:%.*]] = llvm.insertvalue [[rp]], [[undef]][0 : i32] : !llvm.struct<(f32, f32)>
// CHECK: [[full:%.*]] = llvm.insertvalue [[ip]], [[withr]][1 : i32] : !llvm.struct<(f32, f32)>
// CHECK: return [[full]] : !llvm.struct<(f32, f32)>

func.func @test_constc8() -> !fir.complex<8> {
  %0 = fir.constc (#fir.real<8, 1.8>, #fir.real<8, 2.3>) : !fir.complex<8>
  return %0 : !fir.complex<8>
}

// CHECK-LABEL: @test_constc8
// CHECK_SAME: () -> !llvm.struct<(f64, f64)>
// CHECK-DAG: [[rp:%.*]] = llvm.mlir.constant(1.800000e+00 : f64) : f64
// CHECK-DAG: [[ip:%.*]] = llvm.mlir.constant(2.300000e+00 : f64) : f64
// CHECK: [[undef:%.*]] = llvm.mlir.undef : !llvm.struct<(f64, f64)>
// CHECK: [[withr:%.*]] = llvm.insertvalue [[rp]], [[undef]][0 : i32] : !llvm.struct<(f64, f64)>
// CHECK: [[full:%.*]] = llvm.insertvalue [[ip]], [[withr]][1 : i32] : !llvm.struct<(f64, f64)>
// CHECK: return [[full]] : !llvm.struct<(f64, f64)>

// -----

// Test `fir.store` --> `llvm.store` conversion

func.func @test_store_index(%val_to_store : index, %addr : !fir.ref<index>) {
  fir.store %val_to_store to %addr : !fir.ref<index>
  return
}

// CHECK-LABEL:   llvm.func @test_store_index
// CHECK-SAME:    (%[[arg0:.*]]: i64, %[[arg1:.*]]: !llvm.ptr<i64>) {
// CHECK-NEXT:    llvm.store %[[arg0]], %[[arg1]] : !llvm.ptr<i64>
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }

func.func @test_store_box(%array : !fir.ref<!fir.box<!fir.array<?x?xf32>>>, %box : !fir.box<!fir.array<?x?xf32>>) {
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

func.func @test_load_index(%addr : !fir.ref<index>) {
  %0 = fir.load %addr : !fir.ref<index>
  return
}

// CHECK-LABEL: llvm.func @test_load_index(
// CHECK-SAME:  %[[arg1:.*]]: !llvm.ptr<i64>) {
// CHECK-NEXT:    %0 = llvm.load %[[arg1]] : !llvm.ptr<i64>
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }

func.func @test_load_box(%addr : !fir.ref<!fir.box<!fir.array<10xf32>>>) {
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

func.func @extract_rank(%arg0: !fir.box<!fir.array<*:f64>>) -> i32 {
  %0 = fir.box_rank %arg0 : (!fir.box<!fir.array<*:f64>>) -> i32
  return %0 : i32
}

// CHECK-LABEL: llvm.func @extract_rank(
// CHECK-SAME:                          %[[ARG0:.*]]: !llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>) -> i32
// CHECK:         %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[GEP:.*]] = llvm.getelementptr %[[ARG0]][%[[C0]], 3] : (!llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>, i32) -> !llvm.ptr<i32>
// CHECK:         %[[RANK:.*]] = llvm.load %[[GEP]] : !llvm.ptr<i32>
// CHECK:         llvm.return %[[RANK]] : i32

// -----

// Test `fir.box_addr` conversion.

func.func @extract_addr(%arg0: !fir.box<!fir.array<*:f64>>) -> !fir.ref<f64> {
  %0 = fir.box_addr %arg0 : (!fir.box<!fir.array<*:f64>>) -> !fir.ref<f64>
  return %0 : !fir.ref<f64>
}

// CHECK-LABEL: llvm.func @extract_addr(
// CHECK-SAME:                          %[[ARG0:.*]]: !llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>) -> !llvm.ptr<f64>
// CHECK:         %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[GEP:.*]] = llvm.getelementptr %[[ARG0]][%[[C0]], 0] : (!llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>, i32) -> !llvm.ptr<ptr<f64>>
// CHECK:         %[[ADDR:.*]] = llvm.load %[[GEP]] : !llvm.ptr<ptr<f64>>
// CHECK:         llvm.return %[[ADDR]] : !llvm.ptr<f64>

// -----

// Test `fir.box_dims` conversion.

func.func @extract_dims(%arg0: !fir.box<!fir.array<*:f64>>) -> index {
  %c1 = arith.constant 0 : i32
  %cast = fir.convert %arg0 : (!fir.box<!fir.array<*:f64>>) -> !fir.box<!fir.array<?xf64>> 
  %0:3 = fir.box_dims %cast, %c1 : (!fir.box<!fir.array<?xf64>>, i32) -> (index, index, index)
  return %0 : index
}

// CHECK-LABEL: llvm.func @extract_dims(
// CHECK-SAME:                          %[[ARG0:.*]]: !llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>) -> i64
// CHECK:         %[[C0_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:    %[[CAST:.*]] = llvm.bitcast %[[ARG0]] : !llvm.ptr<struct<(ptr<f64>, i64, i32, i8, i8, i8, i8)>> to !llvm.ptr<struct<(ptr<f64>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>>
// CHECK:         %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[C0_2:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[GEP0:.*]] = llvm.getelementptr %[[CAST]][%[[C0]], 7, %[[C0_1]], %[[C0_2]]] :  (!llvm.ptr<struct<(ptr<f64>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>>, i32, i32, i32) -> !llvm.ptr<i64>
// CHECK:         %[[LOAD0:.*]] = llvm.load %[[GEP0]] : !llvm.ptr<i64>
// CHECK:         %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:         %[[GEP1:.*]] = llvm.getelementptr %[[CAST]][%[[C0]], 7, %[[C0_1]], %[[C1]]] : (!llvm.ptr<struct<(ptr<f64>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>>, i32, i32, i32) -> !llvm.ptr<i64>
// CHECK:         %[[LOAD1:.*]] = llvm.load %[[GEP1]] : !llvm.ptr<i64>
// CHECK:         %[[C2:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:         %[[GEP2:.*]] = llvm.getelementptr %[[CAST]][%[[C0]], 7, %[[C0_1]], %[[C2]]] : (!llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>, i32, i32, i32) -> !llvm.ptr<i64>
// CHECK:         %[[LOAD2:.*]] = llvm.load %[[GEP2]] : !llvm.ptr<i64>
// CHECK:         llvm.return %[[LOAD0]] : i64

// -----

// Test `fir.box_elesize` conversion.

func.func @extract_elesize(%arg0: !fir.box<f32>) -> i32 {
  %0 = fir.box_elesize %arg0 : (!fir.box<f32>) -> i32
  return %0 : i32
}

// CHECK-LABEL: llvm.func @extract_elesize(
// CHECK-SAME:                             %[[ARG0:.*]]: !llvm.ptr<struct<(ptr<f32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>) -> i32
// CHECK:         %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[GEP:.*]] = llvm.getelementptr %[[ARG0]][%[[C0]], 1] : (!llvm.ptr<struct<(ptr<f32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>, i32) -> !llvm.ptr<i32>
// CHECK:         %[[ELE_SIZE:.*]] = llvm.load %[[GEP]] : !llvm.ptr<i32>
// CHECK:         llvm.return %[[ELE_SIZE]] : i32

// -----

// Test `fir.box_isarray` conversion.
// `rank` is extracted from `fir.box` and compare to 0.

func.func @box_isarray(%arg0: !fir.box<!fir.array<*:f64>>) -> i1 {
  %0 = fir.box_isarray %arg0 : (!fir.box<!fir.array<*:f64>>) -> i1
  return %0 : i1
}

// CHECK-LABEL: llvm.func @box_isarray(
// CHECK-SAME:                         %[[ARG0:.*]]: !llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>) -> i1
// CHECK:         %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[GEP:.*]] = llvm.getelementptr %[[ARG0]][%[[C0]], 3] : (!llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>, i32) -> !llvm.ptr<i32>
// CHECK:         %[[RANK:.*]] = llvm.load %[[GEP]] : !llvm.ptr<i32>
// CHECK:         %[[C0_ISARRAY:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[IS_ARRAY:.*]] = llvm.icmp "ne" %[[RANK]], %[[C0_ISARRAY]] : i32
// CHECK:         llvm.return %[[IS_ARRAY]] : i1

// -----

// Test `fir.box_isalloc` conversion.
// `attribute` is extracted from `fir.box` and checked against a mask equal to
// the value of `CFI_attribute_allocatable`.

func.func @box_isalloc(%arg0: !fir.box<!fir.array<*:f64>>) -> i1 {
  %0 = fir.box_isalloc %arg0 : (!fir.box<!fir.array<*:f64>>) -> i1
  return %0 : i1
}

// CHECK-LABEL: llvm.func @box_isalloc(
// CHECK-SAME:                         %[[ARG0:.*]]: !llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>) -> i1
// CHECK:         %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[GEP:.*]] = llvm.getelementptr %[[ARG0]][%[[C0]], 5] : (!llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>, i32) -> !llvm.ptr<i32>
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

func.func @box_isptr(%arg0: !fir.box<!fir.array<*:f64>>) -> i1 {
  %0 = fir.box_isptr %arg0 : (!fir.box<!fir.array<*:f64>>) -> i1
  return %0 : i1
}

// CHECK-LABEL: llvm.func @box_isptr(
// CHECK-SAME:                         %[[ARG0:.*]]: !llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>) -> i1
// CHECK:         %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[GEP:.*]] = llvm.getelementptr %[[ARG0]][%[[C0]], 5] : (!llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>, i32) -> !llvm.ptr<i32>
// CHECK:         %[[ATTR:.*]] = llvm.load %[[GEP]] : !llvm.ptr<i32>
// CHECK:         %[[ATTR_ISALLOC:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:         %[[AND:.*]] = llvm.and %[[ATTR]], %[[ATTR_ISALLOC]]  : i32
// CHECK:         %[[CMP_C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[IS_ALLOC:.*]] = llvm.icmp "ne" %[[AND]], %[[CMP_C0]] : i32
// CHECK:         llvm.return %[[IS_ALLOC]] : i1

// -----

// Test fir.alloca of one element

func.func @alloca_one() -> !fir.ref<i32> {
  %1 = fir.alloca i32
  return %1 : !fir.ref<i32>
}

// CHECK-LABEL: llvm.func @alloca_one() -> !llvm.ptr<i32>
// CHECK: [[N:%.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: [[A:%.*]] = llvm.alloca [[N]] x i32
// CHECK: llvm.return [[A]] : !llvm.ptr<i32>

// -----

// Test fir.alloca of several elements

func.func @alloca_several() -> !fir.ref<i32> {
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

func.func @alloca_ptr_to_array() -> !fir.ref<!fir.ptr<!fir.array<?xi32>>> {
  %1 = fir.alloca !fir.ptr<!fir.array<?xi32>>
  return %1 : !fir.ref<!fir.ptr<!fir.array<?xi32>>>
}

// CHECK-LABEL: llvm.func @alloca_ptr_to_array() -> !llvm.ptr<ptr<i32>>
// CHECK: [[ONE:%.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: [[A:%.*]] = llvm.alloca [[ONE]] x !llvm.ptr<i32>
// CHECK: llvm.return [[A]] : !llvm.ptr<ptr<i32>>

// -----

// Test fir.alloca of array of unknown-length chars

func.func @alloca_char_array(%l: i32, %e : index) -> !fir.ref<!fir.array<?x?x!fir.char<1,?>>> {
  %a = fir.alloca !fir.array<?x?x!fir.char<1,?>>(%l : i32), %e, %e
  return %a :  !fir.ref<!fir.array<?x?x!fir.char<1,?>>>
}

// CHECK-LABEL: llvm.func @alloca_char_array
// CHECK-SAME: ([[L:%.*]]: i32, [[E:%.*]]: i64) -> !llvm.ptr<i8>
// CHECK-DAG: [[UNUSEDONE:%.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG: [[LCAST:%.*]] = llvm.sext [[L]] : i32 to i64
// CHECK: [[PROD1:%.*]] = llvm.mul [[LCAST]], [[E]] : i64
// CHECK: [[PROD2:%.*]] = llvm.mul [[PROD1]], [[E]] : i64
// CHECK: [[A:%.*]] = llvm.alloca [[PROD2]] x i8 {in_type = !fir.array<?x?x!fir.char<1,?>>
// CHECK: return [[A]] : !llvm.ptr<i8>

// -----

// Test fir.alloca of array of known-length chars

func.func @alloca_fixed_char_array(%e : index) -> !fir.ref<!fir.array<?x?x!fir.char<1,8>>> {
  %a = fir.alloca !fir.array<?x?x!fir.char<1,8>>, %e, %e
  return %a :  !fir.ref<!fir.array<?x?x!fir.char<1,8>>>
}

// CHECK-LABEL: llvm.func @alloca_fixed_char_array
// CHECK-SAME: ([[E:%.*]]: i64) -> !llvm.ptr<array<8 x i8>>
// CHECK-DAG: [[ONE:%.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: [[PROD1:%.*]] = llvm.mul [[ONE]], [[E]] : i64
// CHECK: [[PROD2:%.*]] = llvm.mul [[PROD1]], [[E]] : i64
// CHECK: [[A:%.*]] = llvm.alloca [[PROD2]] x !llvm.array<8 x i8> {in_type = !fir.array<?x?x!fir.char<1,8>>
// CHECK: return [[A]] : !llvm.ptr<array<8 x i8>>

// -----

// Test fir.alloca of record type with LEN parameters
//   type t(p1,p2)
//      integer, len :: p1
//      integer(kind=2), len :: p2
//      integer f1
//      real f2
//   end type t

func.func private @_QTtP.mem.size(%0 : i32, %1 : i16) -> index

func.func @alloca_record(%arg0 : i32, %arg1 : i16) -> !fir.ref<!fir.type<_QTt(p1:i32,p2:i16){f1:i32,f2:f32}>> {
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

func.func @alloca_multidim_array(%0 : index) -> !fir.ref<!fir.array<8x16x32xf32>> {
  %1 = arith.constant 24 : index
  %2 = fir.alloca !fir.array<8x16x32xf32>, %0, %1
  return %2 : !fir.ref<!fir.array<8x16x32xf32>>
}

// CHECK-LABEL: llvm.func @alloca_multidim_array
// CHECK-SAME: ([[OP1:%.*]]: i64) -> !llvm.ptr<array<32 x array<16 x array<8 x f32>
// CHECK: [[OP2:%.*]] = llvm.mlir.constant(24 : index) : i64
// CHECK: [[ONE:%.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: [[MUL1:%.*]] = llvm.mul [[ONE]], [[OP1]] : i64
// CHECK: [[TOTAL:%.*]] = llvm.mul [[MUL1]], [[OP2]] : i64
// CHECK: [[A:%.*]] = llvm.alloca [[TOTAL]] x !llvm.array<32 x array<16 x array<8 x f32>
// CHECK: llvm.return [[A]] : !llvm.ptr<array<32 x array<16 x array<8 x f32>

// -----

// Test fir.alloca of a multidimensional array with constant interior

func.func @alloca_const_interior_array(%0 : index) -> !fir.ref<!fir.array<8x9x?x?xf32>> {
  %1 = arith.constant 64 : index
  %2 = fir.alloca !fir.array<8x9x?x?xf32>, %0, %1
  return %2 : !fir.ref<!fir.array<8x9x?x?xf32>>
}

// CHECK-LABEL: llvm.func @alloca_const_interior_array
// CHECK-SAME: ([[OP1:%.*]]: i64) -> !llvm.ptr<array<9 x array<8 x f32>
// CHECK: [[OP2:%.*]] = llvm.mlir.constant(64 : index) : i64
// CHECK: [[ONE:%.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: [[MUL1:%.*]] = llvm.mul [[ONE]], [[OP1]] : i64
// CHECK: [[TOTAL:%.*]] = llvm.mul [[MUL1]], [[OP2]] : i64
// CHECK: [[A:%.*]] = llvm.alloca [[TOTAL]] x !llvm.array<9 x array<8 x f32>
// CHECK: llvm.return [[A]] : !llvm.ptr<array<9 x array<8 x f32>

// -----

// Test alloca with an array with holes.
// Constant factor of 60 (4*3*5) must be included.

func.func @alloca_array_with_holes(%0 : index, %1 : index) -> !fir.ref<!fir.array<4x?x3x?x5xi32>> {
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

func.func @select_case_integer(%arg0: !fir.ref<i32>) -> i32 {
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
  cf.br ^bb6
^bb2:  // pred: ^bb0
  %c2_i32_1 = arith.constant 2 : i32
  fir.store %c2_i32_1 to %arg0 : !fir.ref<i32>
  cf.br ^bb6
^bb3:  // pred: ^bb0
  %c0_i32 = arith.constant 0 : i32
  fir.store %c0_i32 to %arg0 : !fir.ref<i32>
  cf.br ^bb6
^bb4:  // pred: ^bb0
  %c4_i32_2 = arith.constant 4 : i32
  fir.store %c4_i32_2 to %arg0 : !fir.ref<i32>
  cf.br ^bb6
^bb5:  // 3 preds: ^bb0, ^bb0, ^bb0
  %c7_i32_3 = arith.constant 7 : i32
  fir.store %c7_i32_3 to %arg0 : !fir.ref<i32>
  cf.br ^bb6
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

func.func @select_case_logical(%arg0: !fir.ref<!fir.logical<4>>) {
  %1 = fir.load %arg0 : !fir.ref<!fir.logical<4>>
  %2 = fir.convert %1 : (!fir.logical<4>) -> i1
  %false = arith.constant false
  %true = arith.constant true
  fir.select_case %2 : i1 [#fir.point, %false, ^bb1,
                           #fir.point, %true, ^bb2,
                            unit, ^bb3]
^bb1:
  %c1_i32 = arith.constant 1 : i32
  cf.br ^bb3
^bb2:
  %c2_i32 = arith.constant 2 : i32
  cf.br ^bb3
^bb3:
  return
}

// CHECK-LABEL: llvm.func @select_case_logical(
// CHECK-SAME:                                 %[[ARG0:.*]]: !llvm.ptr<i32>
// CHECK:         %[[LOAD_ARG0:.*]] = llvm.load %[[ARG0]] : !llvm.ptr<i32>
// CHECK:         %[[CST_ZERO:.*]] = llvm.mlir.constant(0 : i64) : i32
// CHECK:         %[[SELECT_VALUE:.*]] = llvm.icmp "ne" %[[LOAD_ARG0]], %[[CST_ZERO]] : i32
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

func.func @test_is_present_i64(%arg0: !fir.ref<i64>) -> () {
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

func.func @test_is_present_box(%arg0: !fir.box<!fir.ref<i64>>) -> () {
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

func.func @test_absent_i64() -> () {
  %0 = fir.absent !fir.ref<i64>
  return
}

// CHECK-LABEL: @test_absent_i64
// CHECK-NEXT:  %{{.*}} = llvm.mlir.null : !llvm.ptr<i64>
// CHECK-NEXT:  llvm.return
// CHECK-NEXT:  }

func.func @test_absent_box() -> () {
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

func.func @is_present(%arg0: !fir.ref<i64>) -> i1 {
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

func.func @absent() -> i1 {
  %0 = fir.absent !fir.ref<i64>
  %1 = fir.call @is_present(%0) : (!fir.ref<i64>) -> i1
  return %1 : i1
}

// CHECK-LABEL: @absent
// CHECK-SAME:  () -> i1
// CHECK-NEXT:  %[[ptr:.*]] = llvm.mlir.null : !llvm.ptr<i64>
// CHECK-NEXT:  %[[ret_val:.*]] = llvm.call @is_present(%[[ptr]]) : (!llvm.ptr<i64>) -> i1
// CHECK-NEXT:  llvm.return %[[ret_val]] : i1

// -----

// Test `fir.string_lit` conversion.

func.func @string_lit0() {
  %1 = fir.string_lit "Hello, World!"(13) : !fir.char<1>
  return
}

// CHECK-LABEL: llvm.func @string_lit0
// CHECK: %{{.*}} = llvm.mlir.constant("Hello, World!") : !llvm.array<13 x i8>

func.func @string_lit1() {
  %2 = fir.string_lit [158, 2345](2) : !fir.char<2>
  return
}

// CHECK-LABEL: llvm.func @string_lit1
// %{{.*}} = llvm.mlir.constant(dense<[158, 2345]> : vector<2xi16>) : !llvm.array<2 x i16>

// -----

// Test must be dead conversion.

func.func @dead_shift() {
  %c0 = arith.constant 0 : index
  %0 = fir.shift %c0 : (index) -> !fir.shift<1>
  return
}

// CHECK-LABEL: llvm.func @dead_shift
// CHECK-NOT: fir.shift
// CHECK: %{{.*}} = llvm.mlir.constant(0 : index) : i{{.*}}
// CHECK-NEXT: llvm.return

func.func @dead_shape() {
  %c0 = arith.constant 0 : index
  %0 = fir.shape %c0 : (index) -> !fir.shape<1>
  return
}

// CHECK-LABEL: llvm.func @dead_shape
// CHECK-NOT: fir.shape
// CHECK: %{{.*}} = llvm.mlir.constant(0 : index) : i{{.*}}
// CHECK-NEXT: llvm.return

func.func @dead_shapeshift() {
  %c0 = arith.constant 0 : index
  %0 = fir.shape_shift %c0, %c0 : (index, index) -> !fir.shapeshift<1>
  return
}

// CHECK-LABEL: llvm.func @dead_shapeshift
// CHECK-NOT: fir.shape_shift
// CHECK: %{{.*}} = llvm.mlir.constant(0 : index) : i{{.*}}
// CHECK-NEXT: llvm.return

func.func @dead_slice() {
  %c0 = arith.constant 0 : index
  %0 = fir.slice %c0, %c0, %c0 : (index, index, index) -> !fir.slice<1>
  return
}

// CHECK-LABEL: llvm.func @dead_slice
// CHECK-NOT: fir.slice
// CHECK: %{{.*}} = llvm.mlir.constant(0 : index) : i{{.*}}
// CHECK-NEXT: llvm.return

// -----

// Test `fir.box_tdesc` conversion.

func.func @box_tdesc(%arg0: !fir.box<f64>) {
  %0 = fir.box_tdesc %arg0 : (!fir.box<f64>) -> !fir.tdesc<f64>
  return
}

// CHECK-LABEL: llvm.func @box_tdesc(
// CHECK-SAME:                       %[[ARG0:.*]]: !llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>) {
// CHECK:         %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[GEP:.*]] = llvm.getelementptr %[[ARG0]][%[[C0]], 4] : (!llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>, i32) -> !llvm.ptr<i8>
// CHECK:         %[[LOAD:.*]] = llvm.load %[[GEP]] : !llvm.ptr<i{{.*}}>
// CHECK:         %{{.*}} = llvm.inttoptr %[[LOAD]] : i{{.*}} to !llvm.ptr<i{{.*}}>

// -----

// Test `fir.embox` conversion.

// Check basic creation of a descriptor and insertion of values.
// The indices used to insert values into the descriptor correspond the
// position of the fields in the descriptor as defined in `CFI_cdesc_t` in
// flang/ISO_Fortran_binding.h.

func.func @embox0(%arg0: !fir.ref<!fir.array<100xi32>>) {
  %0 = fir.embox %arg0() : (!fir.ref<!fir.array<100xi32>>) -> !fir.box<!fir.array<100xi32>>
  return
}

// CHECK-LABEL: func @embox0(
// CHECK-SAME:               %[[ARG0:.*]]: !llvm.ptr<array<100 x i32>>
// CHECK:         %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:         %[[ALLOCA:.*]] = llvm.alloca %[[C1]] x !llvm.struct<(ptr<array<100 x i32>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})> {alignment = 8 : i64} : (i32) -> !llvm.ptr<struct<(ptr<array<100 x i32>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>
// CHECK:         %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<array<100 x i32>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>
// CHECK:         %[[ELEM_SIZE:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK:         %[[TYPE_CODE:.*]] = llvm.mlir.constant(9 : i32) : i32
// CHECK:         %[[I64_ELEM_SIZE:.*]] = llvm.sext %[[ELEM_SIZE]] : i32 to i64
// CHECK:         %[[DESC0:.*]] = llvm.insertvalue %[[I64_ELEM_SIZE]], %[[DESC]][1 : i32] : !llvm.struct<(ptr<array<100 x i32>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>
// CHECK:         %[[CFI_VERSION:.*]] = llvm.mlir.constant(20180515 : i32) : i32
// CHECK:         %[[DESC1:.*]] = llvm.insertvalue %[[CFI_VERSION]], %[[DESC0]][2 : i32] : !llvm.struct<(ptr<array<100 x i32>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>
// CHECK:         %[[RANK:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[RANK_I8:.*]] = llvm.trunc %[[RANK]] : i32 to i8
// CHECK:         %[[DESC2:.*]] = llvm.insertvalue %[[RANK_I8]], %[[DESC1]][3 : i32] : !llvm.struct<(ptr<array<100 x i32>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>
// CHECK:         %[[TYPE_CODE_I8:.*]] = llvm.trunc %[[TYPE_CODE]] : i32 to i8
// CHECK:         %[[DESC3:.*]] = llvm.insertvalue %[[TYPE_CODE_I8]], %[[DESC2]][4 : i32] : !llvm.struct<(ptr<array<100 x i32>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>
// CHECK:         %[[ATTR:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[ATTR_I8:.*]] = llvm.trunc %[[ATTR]] : i32 to i8
// CHECK:         %[[DESC4:.*]] = llvm.insertvalue %[[ATTR_I8]], %[[DESC3]][5 : i32] : !llvm.struct<(ptr<array<100 x i32>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>
// CHECK:         %[[F18ADDENDUM:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[F18ADDENDUM_I8:.*]] = llvm.trunc %[[F18ADDENDUM]] : i32 to i8
// CHECK:         %[[DESC5:.*]] = llvm.insertvalue %[[F18ADDENDUM_I8]], %[[DESC4]][6 : i32] : !llvm.struct<(ptr<array<100 x i32>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>
// CHECK:         %[[ADDR:.*]] = llvm.bitcast %[[ARG0]] : !llvm.ptr<array<100 x i32>> to !llvm.ptr<array<100 x i32>>
// CHECK:         %[[DESC6:.*]] = llvm.insertvalue %[[ADDR]], %[[DESC5]][0 : i32] : !llvm.struct<(ptr<array<100 x i32>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>
// CHECK:         llvm.store %[[DESC6]], %[[ALLOCA]] : !llvm.ptr<struct<(ptr<array<100 x i32>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>

// Check `fir.embox` in a `fir.global`. Descriptors created by `fir.embox`
// conversion are not generating `alloca` instructions. This test make sure of
// that.

fir.global @box_global : !fir.ref<!fir.array<?xi32>> {
  %arr = fir.zero_bits !fir.ref<!fir.array<?xi32>>
  %0 = arith.constant 0 : index
  %3 = fir.embox %arr: (!fir.ref<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
  fir.has_value %arr : !fir.ref<!fir.array<?xi32>>
}

// CHECK-LABEL: llvm.mlir.global external @box_global
// CHECK-NOT:     llvm.alloca

// Check `fir.embox` conversion of a POINTER entity. Make sure that the
// attribute in the descriptor is set to 1 (value of CFI_attribute_pointer
// in flang/ISO_Fortran_binding.h).

func.func @embox_pointer(%arg0: !fir.ref<i32>) {
  %0 = fir.embox %arg0 : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
  return
}

// CHECK-LABEL: llvm.func @embox_pointer
// Check 1st 1 constant to skip it.
// CHECK: %{{.*}} = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[CFI_ATTR_POINTER:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[ATTR_I8:.*]] = llvm.trunc %[[CFI_ATTR_POINTER]] : i32 to i8
// CHECK: %{{.*}} = llvm.insertvalue %[[ATTR_I8]], %{{.*}}[5 : i32] : !llvm.struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>

// Check `fir.embox` conversion of an ALLOCATABLE entity. Make sure that the
// attribute in the descriptor is set to 2 (value of CFI_attribute_allocatable
// in flang/ISO_Fortran_binding.h).

func.func @embox_allocatable(%arg0: !fir.heap<!fir.array<?x!fir.char<1,10>>>) {
  %0 = fir.embox %arg0 : (!fir.heap<!fir.array<?x!fir.char<1,10>>>) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>
  return
}

// CHECK-LABEL: llvm.func @embox_allocatable
// CHECK: %[[CFI_ATTR_ALLOCATABLE:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK: %[[ATTR_I8:.*]] = llvm.trunc %[[CFI_ATTR_ALLOCATABLE]] : i32 to i8
// CHECK: %{{.*}} = llvm.insertvalue %[[ATTR_I8]], %{{.*}}[5 : i32] : !llvm.struct<(ptr<array<10 x i8>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>

// Check `fir.embox` conversion of a type code.

func.func @embox_typecode0(%arg0: !fir.ref<i64>) {
  %0 = fir.embox %arg0 : (!fir.ref<i64>) -> !fir.box<!fir.ptr<i64>>
  return
}

// CHECK-LABEL: llvm.func @embox_typecode0
// CHECK: %[[TYPE_CODE_I64:.*]] = llvm.mlir.constant(10 : i32) : i32
// CHECK: %[[TYPE_CODE_I64_I8:.*]] = llvm.trunc %[[TYPE_CODE_I64]] : i32 to i8
// CHECK: %{{.*}} = llvm.insertvalue %[[TYPE_CODE_I64_I8]], %{{.*}}[4 : i32] : !llvm.struct<(ptr<i64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>

func.func @embox_typecode1(%arg0: !fir.ref<f32>) {
  %0 = fir.embox %arg0 : (!fir.ref<f32>) -> !fir.box<!fir.ptr<f32>>
  return
}

// CHECK-LABEL: llvm.func @embox_typecode1
// CHECK: %[[TYPE_CODE_F32:.*]] = llvm.mlir.constant(27 : i32) : i32
// CHECK: %[[TYPE_CODE_F32_I8:.*]] = llvm.trunc %[[TYPE_CODE_I64]] : i32 to i8
// CHECK: %{{.*}} = llvm.insertvalue %[[TYPE_CODE_F32_I8]], %{{.*}}[4 : i32] : !llvm.struct<(ptr<f32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>

func.func @embox_typecode2(%arg0: !fir.ref<f128>) {
  %0 = fir.embox %arg0 : (!fir.ref<f128>) -> !fir.box<!fir.ptr<f128>>
  return
}

// CHECK-LABEL: llvm.func @embox_typecode2
// CHECK: %[[TYPE_CODE_F128:.*]] = llvm.mlir.constant(31 : i32) : i32
// CHECK: %[[TYPE_CODE_F128_I8:.*]] = llvm.trunc %[[TYPE_CODE_F128]] : i32 to i8
// CHECK: %{{.*}} = llvm.insertvalue %[[TYPE_CODE_F128_I8]], %{{.*}}[4 : i32] : !llvm.struct<(ptr<f128>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>

func.func @embox_typecode3(%arg0: !fir.ref<!fir.complex<4>>) {
  %0 = fir.embox %arg0 : (!fir.ref<!fir.complex<4>>) -> !fir.box<!fir.ptr<!fir.complex<4>>>
  return
}

// CHECK-LABEL: llvm.func @embox_typecode3
// CHECK: %[[TYPE_CODE_CPLX4:.*]] = llvm.mlir.constant(34 : i32) : i32
// CHECK: %[[TYPE_CODE_CPLX4_I8:.*]] = llvm.trunc %[[TYPE_CODE_F128]] : i32 to i8
// CHECK: %{{.*}} = llvm.insertvalue %[[TYPE_CODE_CPLX4_I8]], %{{.*}}[4 : i32] : !llvm.struct<(ptr<struct<(f32, f32)>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>

func.func @embox_typecode4(%arg0: !fir.ref<!fir.logical<1>>) {
  %0 = fir.embox %arg0 : (!fir.ref<!fir.logical<1>>) -> !fir.box<!fir.ptr<!fir.logical<1>>>
  return
}

// CHECK-LABEL: llvm.func @embox_typecode4
// CHECK: %[[TYPE_CODE_I64:.*]] = llvm.mlir.constant(39 : i32) : i32
// CHECK: %[[TYPE_CODE_I64_I8:.*]] = llvm.trunc %[[TYPE_CODE_I64]] : i32 to i8
// CHECK: %{{.*}} = llvm.insertvalue %[[TYPE_CODE_I64_I8]], %{{.*}}[4 : i32] : !llvm.struct<(ptr<i8>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>

// -----

// Test `fir.embox` conversion. This test creates a global so it needs to be
// split from others.

// Check descriptor for a derived type. Check that the f18Addendum flag is set
// to 1 meaning the addendum is present (true) and the addendum values are
// inserted.

fir.global linkonce @_QMtest_dinitE.dt.tseq constant : i8

func.func @embox1(%arg0: !fir.ref<!fir.type<_QMtest_dinitTtseq{i:i32}>>) {
  %0 = fir.embox %arg0() : (!fir.ref<!fir.type<_QMtest_dinitTtseq{i:i32}>>) -> !fir.box<!fir.type<_QMtest_dinitTtseq{i:i32}>>
  return
}

// CHECK: llvm.mlir.global linkonce constant @_QMtest_dinitE.dt.tseq() : i8
// CHECK-LABEL: llvm.func @embox1
// CHECK:         %[[TYPE_CODE:.*]] = llvm.mlir.constant(42 : i32) : i32
// CHECK:         %[[TYPE_CODE_I8:.*]] = llvm.trunc %[[TYPE_CODE]] : i32 to i8
// CHECK:         %{{.*}} = llvm.insertvalue %[[TYPE_CODE_I8]], %{{.*}}[4 : i32] : !llvm.struct<(ptr<struct<"_QMtest_dinitTtseq", (i32)>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, ptr<i{{.*}}>, array<1 x i{{.*}}>)>
// CHECK:         %[[F18ADDENDUM:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:         %[[F18ADDENDUM_I8:.*]] = llvm.trunc %[[F18ADDENDUM]] : i32 to i8
// CHECK:         %{{.*}} = llvm.insertvalue %[[F18ADDENDUM_I8]], %18[6 : i32] : !llvm.struct<(ptr<struct<"_QMtest_dinitTtseq", (i32)>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, ptr<i{{.*}}>, array<1 x i{{.*}}>)>
// CHECK:         %[[TDESC:.*]] = llvm.mlir.addressof @_QMtest_dinitE.dt.tseq : !llvm.ptr<i8>
// CHECK:         %[[TDESC_CAST:.*]] = llvm.bitcast %22 : !llvm.ptr<i8> to !llvm.ptr<i8>
// CHECK:         %{{.*}} = llvm.insertvalue %[[TDESC_CAST]], %{{.*}}[7 : i32] : !llvm.struct<(ptr<struct<"_QMtest_dinitTtseq", (i32)>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, ptr<i{{.*}}>, array<1 x i{{.*}}>)>

// -----

// Test `fir.field_index`

func.func @field_index_static_size_1_elem() -> () {
  %1 = fir.field_index i, !fir.type<t1{i:i32}>
  return
}

// CHECK-LABEL: @field_index_static_size_1_elem
// CHECK-NEXT:  %{{.*}} = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:  llvm.return

func.func @field_index_static_size_3_elems() -> () {
  %1 = fir.field_index k, !fir.type<t2{i:i32, j:f32, k:i8}>
  return
}

// CHECK-LABEL: @field_index_static_size_3_elems
// CHECK-NEXT:  %{{.*}} = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:  llvm.return

//  When converting `fir.field_index` for a dynamically sized record, the
//  offset will be calculated at runtime by calling methods like the ones
//  below. Note that these methods would normally be generated by the compiler.
func.func private @custom_typeP.field_1.offset() -> i32
func.func private @custom_typeP.field_2.offset() -> i32

func.func @field_index_dynamic_size() -> () {
  %1 = fir.field_index field_1, !fir.type<custom_type{field_1:i32, field_2:!fir.array<?xf32>}>
  %2 = fir.field_index field_2, !fir.type<custom_type{field_1:i32, field_2:!fir.array<?xf32>}>
  return
}

// CHECK-LABEL: @field_index_dynamic_size
// CHECK-NEXT: %{{.*}} = llvm.call @custom_typeP.field_1.offset() {field = 0 : i64} : () -> i32
// CHECK-NEXT: %{{.*}} = llvm.call @custom_typeP.field_2.offset() {field = 1 : i64} : () -> i32
// CHECK-NEXT:  llvm.return

// -----

// Check `fir.no_reassoc` conversion to LLVM IR dialect

func.func @no_reassoc(%arg0: !fir.ref<i32>) {
  %0 = fir.alloca i32
  %1 = fir.load %arg0 : !fir.ref<i32>
  %2 = fir.no_reassoc %1 : i32
  fir.store %2 to %0 : !fir.ref<i32>
  return
}

// CHECK-LABEL: llvm.func @no_reassoc(
// CHECK-SAME:                        %[[ARG0:.*]]: !llvm.ptr<i32>) {
// CHECK:         %[[C1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[ALLOC:.*]] = llvm.alloca %[[C1]] x i32 {in_type = i32, operand_segment_sizes = dense<0> : vector<2xi32>} : (i64) -> !llvm.ptr<i32>
// CHECK:         %[[LOAD:.*]] = llvm.load %[[ARG0]] : !llvm.ptr<i32>
// CHECK:         llvm.store %[[LOAD]], %[[ALLOC]] : !llvm.ptr<i32>
// CHECK:         llvm.return

// -----

// Test `fircg.ext_embox` conversion.

// Check complete `fircg.ext_embox`.

func.func @xembox0(%arg0: !fir.ref<!fir.array<?xi32>>) {
  %c0 = arith.constant 0 : i64
  %0 = fircg.ext_embox %arg0(%c0) origin %c0[%c0, %c0, %c0] : (!fir.ref<!fir.array<?xi32>>, i64, i64, i64, i64, i64) -> !fir.box<!fir.array<?xi32>>
  return
}

// CHECK-LABEL: llvm.func @xembox0(
// CHECK-SAME:                     %[[ARG0:.*]]: !llvm.ptr<i32>
// CHECK:         %[[ALLOCA_SIZE:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:         %[[ALLOCA:.*]] = llvm.alloca %[[ALLOCA_SIZE]] x !llvm.struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr<struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>>
// CHECK:         %[[C0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[BOX0:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>
// CHECK:         %[[ELEM_LEN:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK:         %[[TYPE:.*]] = llvm.mlir.constant(9 : i32) : i32
// CHECK:         %[[ELEM_LEN_I64:.*]] = llvm.sext %[[ELEM_LEN]] : i32 to i64
// CHECK:         %[[BOX1:.*]] = llvm.insertvalue %[[ELEM_LEN_I64]], %[[BOX0]][1 : i32] : !llvm.struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>
// CHECK:         %[[VERSION:.*]] = llvm.mlir.constant(20180515 : i32) : i32
// CHECK:         %[[BOX2:.*]] = llvm.insertvalue %[[VERSION]], %[[BOX1]][2 : i32] : !llvm.struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>
// CHECK:         %[[RANK:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:         %[[RANK_I8:.*]] = llvm.trunc %[[RANK]] : i32 to i8
// CHECK:         %[[BOX3:.*]] = llvm.insertvalue %[[RANK_I8]], %[[BOX2]][3 : i32] : !llvm.struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>
// CHECK:         %[[TYPE_I8:.*]] = llvm.trunc %[[TYPE]] : i32 to i8
// CHECK:         %[[BOX4:.*]] = llvm.insertvalue %[[TYPE_I8]], %[[BOX3]][4 : i32] : !llvm.struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>
// CHECK:         %[[ATTR:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[ATTR_I8:.*]] = llvm.trunc %[[ATTR]] : i32 to i8
// CHECK:         %[[BOX5:.*]] = llvm.insertvalue %[[ATTR_I8]], %[[BOX4]][5 : i32] : !llvm.struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>
// CHECK:         %[[F18ADDENDUM:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[F18ADDENDUM_I8:.*]] = llvm.trunc %[[F18ADDENDUM]] : i32 to i8
// CHECK:         %[[BOX6:.*]] = llvm.insertvalue %[[F18ADDENDUM_I8]], %[[BOX5]][6 : i32] : !llvm.struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>
// CHECK:         %[[ZERO:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[ELEM_LEN_I64:.*]] = llvm.sext %[[ELEM_LEN]] : i32 to i64
// CHECK:         %[[ADJUSTED_OFFSET:.*]] = llvm.sub %[[C0]], %[[C0]]  : i64
// CHECK:         %[[DIM_OFFSET:.*]] = llvm.mul %[[ADJUSTED_OFFSET]], %[[ONE]]  : i64
// CHECK:         %[[PTR_OFFSET:.*]] = llvm.add %[[DIM_OFFSET]], %[[ZERO]]  : i64
// CHECK:         %[[EXTENT0:.*]] = llvm.sub %[[C0]], %[[C0]]  : i64
// CHECK:         %[[EXTENT1:.*]] = llvm.add %[[EXTENT0]], %[[C0]]  : i64
// CHECK:         %[[EXTENT2:.*]] = llvm.sdiv %[[EXTENT1]], %[[C0]]  : i64
// CHECK:         %[[EXTENT_CMP:.*]] = llvm.icmp "sgt" %[[EXTENT2]], %[[ZERO]] : i64
// CHECK:         %[[EXTENT:.*]] = llvm.select %[[EXTENT_CMP]], %[[EXTENT2]], %[[ZERO]] : i1, i64
// CHECK:         %[[BOX7:.*]] = llvm.insertvalue %[[ONE]], %[[BOX6]][7 : i32, 0 : i32, 0 : i32] : !llvm.struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>
// CHECK:         %[[BOX8:.*]] = llvm.insertvalue %[[EXTENT]], %[[BOX7]][7 : i32, 0 : i32, 1 : i32] : !llvm.struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>
// CHECK:         %[[STRIDE:.*]] = llvm.mul %[[ELEM_LEN_I64]], %[[C0]]  : i64
// CHECK:         %[[BOX9:.*]] = llvm.insertvalue %[[STRIDE]], %[[BOX8]][7 : i32, 0 : i32, 2 : i32] : !llvm.struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>
// CHECK:         %[[PREV_DIM:.*]] = llvm.mul %[[ELEM_LEN_I64]], %[[C0]]  : i64
// CHECK:         %[[PREV_PTROFF:.*]] = llvm.mul %[[ONE]], %[[C0]]  : i64
// CHECK:         %[[BASE_PTR:.*]] = llvm.getelementptr %[[ARG0]][%[[PTR_OFFSET]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
// CHECK:         %[[ADDR_BITCAST:.*]] = llvm.bitcast %[[BASE_PTR]] : !llvm.ptr<i32> to !llvm.ptr<i32>
// CHECK:         %[[BOX10:.*]] = llvm.insertvalue %[[ADDR_BITCAST]], %[[BOX9]][0 : i32] : !llvm.struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>
// CHECK:         llvm.store %[[BOX10]], %[[ALLOCA]] : !llvm.ptr<struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>>

// Check adjustment of element scaling factor.

func.func @xembox1(%arg0: !fir.ref<!fir.array<?x!fir.char<1, 10>>>) {
  %c0 = arith.constant 0 : i64
  %0 = fircg.ext_embox %arg0(%c0) origin %c0[%c0, %c0, %c0] : (!fir.ref<!fir.array<?x!fir.char<1, 10>>>, i64, i64, i64, i64, i64) -> !fir.box<!fir.array<?x!fir.char<1, 10>>>
  return
}

// CHECK-LABEL: llvm.func @xembox1(%{{.*}}: !llvm.ptr<array<10 x i8>>) {
// CHECK:         %[[C0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[ELEM_LEN:.*]] = llvm.mlir.constant(10 : i32) : i32
// CHECK:         %[[ELEM_LEN_I64:.*]] = llvm.sext %[[ELEM_LEN]] : i32 to i64
// CHECK:         %{{.*}} = llvm.insertvalue %[[ELEM_LEN_I64]], %{{.*}}[1 : i32] : !llvm.struct<(ptr<array<10 x i8>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>
// CHECK:         %[[PTR_OFFSET:.*]] = llvm.sext %[[ELEM_LEN]] : i32 to i64
// CHECK:         %[[PREV_PTROFF:.*]] = llvm.mul %[[PTR_OFFSET]], %[[C0]]  : i64

// Fortran realistic use case extracted from the following snippet:
//
// ```
// subroutine sb(n,sh1,sh2)
//   integer::n,sh1,sh2
//   double precision::arr(sh1:n,sh2:n)
//   call xb(arr(2:n,4:n))
// end subroutine 
// ```

// N is the upperbound, sh1 and sh2 are the shifts or lowerbounds
func.func @_QPsb(%N: index, %sh1: index, %sh2: index) {
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // Calculate nelems in dim1
  %n1_tmp = arith.subi %N, %sh1 : index
  %n1 = arith.addi %n1_tmp, %c1 : index
  // Calculate nelems in dim2
  %n2_tmp = arith.subi %N, %sh2 : index
  %n2 = arith.addi %n2_tmp, %c1 : index
  %arr = fir.alloca !fir.array<?x?xf64>, %n1, %n2 {bindc_name = "arr", uniq_name = "_QFsbEarr"}
  %box = fircg.ext_embox %arr(%n1, %n2) origin %sh1, %sh2[%c2, %N, %c1, %c4, %N, %c1] : (!fir.ref<!fir.array<?x?xf64>>, index, index, index, index, index, index, index, index, index, index) -> !fir.box<!fir.array<?x?xf64>>
  fir.call @_QPxb(%box) : (!fir.box<!fir.array<?x?xf64>>) -> ()
  return
}      
func.func private @_QPxb(!fir.box<!fir.array<?x?xf64>>)

// CHECK-LABEL: llvm.func @_QPsb(
// CHECK-SAME:                   %[[N:.*]]: i64, %[[SH1:.*]]: i64, %[[SH2:.*]]: i64) {
// CHECK:         %[[ALLOCA_SIZE:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:         %[[ALLOCA:.*]] = llvm.alloca %[[ALLOCA_SIZE]] x !llvm.struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<2 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<2 x array<3 x i64>>)>>
// CHECK:         %[[C4:.*]] = llvm.mlir.constant(4 : index) : i64
// CHECK:         %[[C1:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:         %[[C2:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK:         %[[N1_TMP:.*]] = llvm.sub %[[N]], %[[SH1]]  : i64
// CHECK:         %[[N1:.*]] = llvm.add %[[N1_TMP]], %[[C1]]  : i64
// CHECK:         %[[N2_TMP:.*]] = llvm.sub %[[N]], %[[SH2]]  : i64
// CHECK:         %[[N2:.*]] = llvm.add %[[N2_TMP]], %[[C1]]  : i64
// CHECK:         %[[C1_0:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[ARR_SIZE_TMP1:.*]] = llvm.mul %[[C1_0]], %[[N1]]  : i64
// CHECK:         %[[ARR_SIZE:.*]] = llvm.mul %[[ARR_SIZE_TMP1]], %[[N2]]  : i64
// CHECK:         %[[ARR:.*]] = llvm.alloca %[[ARR_SIZE]] x f64 {bindc_name = "arr", in_type = !fir.array<?x?xf64>, operand_segment_sizes = dense<[0, 2]> : vector<2xi32>, uniq_name = "_QFsbEarr"} : (i64) -> !llvm.ptr<f64>
// CHECK:         %[[BOX0:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<2 x array<3 x i64>>)>
// CHECK:         %[[ELEM_LEN:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK:         %[[TYPE_CODE:.*]] = llvm.mlir.constant(28 : i32) : i32
// CHECK:         %[[ELEM_LEN_I64:.*]] = llvm.sext %[[ELEM_LEN]] : i32 to i64
// CHECK:         %[[BOX1:.*]] = llvm.insertvalue %[[ELEM_LEN_I64]], %[[BOX0]][1 : i32] : !llvm.struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<2 x array<3 x i64>>)>
// CHECK:         %[[VERSION:.*]] = llvm.mlir.constant(20180515 : i32) : i32
// CHECK:         %[[BOX2:.*]] = llvm.insertvalue %[[VERSION]], %[[BOX1]][2 : i32] : !llvm.struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<2 x array<3 x i64>>)>
// CHECK:         %[[RANK:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:         %[[RANK_I8:.*]] = llvm.trunc %[[RANK]] : i32 to i8
// CHECK:         %[[BOX3:.*]] = llvm.insertvalue %[[RANK_I8]], %[[BOX2]][3 : i32] : !llvm.struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<2 x array<3 x i64>>)>
// CHECK:         %[[TYPE_I8:.*]] = llvm.trunc %[[TYPE_CODE]] : i32 to i8
// CHECK:         %[[BOX4:.*]] = llvm.insertvalue %[[TYPE_I8]], %[[BOX3]][4 : i32] : !llvm.struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<2 x array<3 x i64>>)>
// CHECK:         %[[ATTR:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[ATTR_I8:.*]] = llvm.trunc %[[ATTR]] : i32 to i8
// CHECK:         %[[BOX5:.*]] = llvm.insertvalue %[[ATTR_I8]], %[[BOX4]][5 : i32] : !llvm.struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<2 x array<3 x i64>>)>
// CHECK:         %[[F18ADDENDUM:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[F18ADDENDUM_I8:.*]] = llvm.trunc %[[F18ADDENDUM]] : i32 to i8
// CHECK:         %[[BOX6:.*]] = llvm.insertvalue %[[F18ADDENDUM_I8]], %[[BOX5]][6 : i32] : !llvm.struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<2 x array<3 x i64>>)>
// CHECK:         %[[ZERO:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[ELEM_LEN_I64:.*]] = llvm.sext %[[ELEM_LEN]] : i32 to i64
// CHECK:         %[[ADJUSTED_OFFSET:.*]] = llvm.sub %[[C2]], %[[SH1]] : i64
// CHECK:         %[[DIM_OFFSET:.*]] = llvm.mul %[[ADJUSTED_OFFSET]], %[[ONE]]  : i64
// CHECK:         %[[PTR_OFFSET:.*]] = llvm.add %[[DIM_OFFSET]], %[[ZERO]]  : i64
// CHECK:         %[[EXTENT0:.*]] = llvm.sub %[[ARG0]], %[[C2]]  : i64
// CHECK:         %[[EXTENT1:.*]] = llvm.add %[[EXTENT0]], %[[C1]]  : i64
// CHECK:         %[[EXTENT2:.*]] = llvm.sdiv %[[EXTENT1]], %[[C1]]  : i64
// CHECK:         %[[EXTENT_CMP:.*]] = llvm.icmp "sgt" %[[EXTENT2]], %[[ZERO]] : i64
// CHECK:         %[[EXTENT:.*]] = llvm.select %[[EXTENT_CMP]], %[[EXTENT2]], %[[ZERO]] : i1, i64
// CHECK:         %[[BOX7:.*]] = llvm.insertvalue %[[ONE]], %[[BOX6]][7 : i32, 0 : i32, 0 : i32] : !llvm.struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<2 x array<3 x i64>>)>
// CHECK:         %[[BOX8:.*]] = llvm.insertvalue %[[EXTENT]], %[[BOX7]][7 : i32, 0 : i32, 1 : i32] : !llvm.struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<2 x array<3 x i64>>)>
// CHECK:         %[[STRIDE:.*]] = llvm.mul %[[ELEM_LEN_I64]], %[[C1]]  : i64
// CHECK:         %[[BOX9:.*]] = llvm.insertvalue %[[STRIDE]], %[[BOX8]][7 : i32, 0 : i32, 2 : i32] : !llvm.struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<2 x array<3 x i64>>)>
// CHECK:         %[[PREV_DIM:.*]] = llvm.mul %[[ELEM_LEN_I64]], %[[N1]]  : i64
// CHECK:         %[[PREV_PTROFF:.*]] = llvm.mul %[[ONE]], %[[N1]]  : i64
// CHECK:         %[[ADJUSTED_OFFSET:.*]] = llvm.sub %[[C4]], %[[SH2]]  : i64
// CHECK:         %[[DIM_OFFSET:.*]] = llvm.mul %[[ADJUSTED_OFFSET]], %[[PREV_PTROFF]]  : i64
// CHECK:         %[[PTR_OFFSET0:.*]] = llvm.add %[[DIM_OFFSET]], %[[PTR_OFFSET]]  : i64
// CHECK:         %[[EXT_SUB:.*]] = llvm.sub %[[N]], %[[C4]]  : i64
// CHECK:         %[[EXT_ADD:.*]] = llvm.add %[[EXT_SUB]], %[[C1]]  : i64
// CHECK:         %[[EXT_SDIV:.*]] = llvm.sdiv %[[EXT_ADD]], %[[C1]]  : i64
// CHECK:         %[[EXT_ICMP:.*]] = llvm.icmp "sgt" %[[EXT_SDIV]], %[[ZERO]] : i64
// CHECK:         %[[EXT_SELECT:.*]] = llvm.select %[[EXT_ICMP]], %[[EXT_SDIV]], %[[ZERO]] : i1, i64
// CHECK:         %[[BOX10:.*]] = llvm.insertvalue %[[ONE]], %[[BOX9]][7 : i32, 1 : i32, 0 : i32] : !llvm.struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<2 x array<3 x i64>>)>
// CHECK:         %[[BOX11:.*]] = llvm.insertvalue %[[EXT_SELECT]], %[[BOX10]][7 : i32, 1 : i32, 1 : i32] : !llvm.struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<2 x array<3 x i64>>)>
// CHECK:         %[[STRIDE_MUL:.*]] = llvm.mul %[[PREV_DIM]], %[[C1]]  : i64
// CHECK:         %[[BOX12:.*]] = llvm.insertvalue %[[STRIDE_MUL]], %[[BOX11]][7 : i32, 1 : i32, 2 : i32] : !llvm.struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<2 x array<3 x i64>>)>
// CHECK:         %[[BASE_PTR:.*]] = llvm.getelementptr %[[ARR]][%[[PTR_OFFSET0]]] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
// CHECK:         %[[ADDR_BITCAST:.*]] = llvm.bitcast %[[BASE_PTR]] : !llvm.ptr<f64> to !llvm.ptr<f64>
// CHECK:         %[[BOX13:.*]] = llvm.insertvalue %[[ADDR_BITCAST]], %[[BOX12]][0 : i32] : !llvm.struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<2 x array<3 x i64>>)>
// CHECK:         llvm.store %[[BOX13]], %[[ALLOCA]] : !llvm.ptr<struct<(ptr<f64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<2 x array<3 x i64>>)>>

// Conversion with a subcomponent.

func.func @_QPtest_dt_slice() {
  %c20 = arith.constant 20 : index
  %c1_i64 = arith.constant 1 : i64
  %c10_i64 = arith.constant 10 : i64
  %c2_i64 = arith.constant 2 : i64
  %0 = fir.alloca i32 {bindc_name = "v", uniq_name = "_QFtest_dt_sliceEv"}
  %1 = fir.alloca !fir.array<20x!fir.type<_QFtest_dt_sliceTt{i:i32,j:i32}>> {bindc_name = "x", uniq_name = "_QFtest_dt_sliceEx"}
  %2 = fir.field_index i, !fir.type<_QFtest_dt_sliceTt{i:i32,j:i32}>
  %5 = fircg.ext_embox %1(%c20)[%c1_i64, %c10_i64, %c2_i64] path %2 : (!fir.ref<!fir.array<20x!fir.type<_QFtest_dt_sliceTt{i:i32,j:i32}>>>, index, i64, i64, i64, !fir.field) -> !fir.box<!fir.array<?xi32>>
  fir.call @_QPtest_dt_callee(%5) : (!fir.box<!fir.array<?xi32>>) -> ()
  return
}
func.func private @_QPtest_dt_callee(%arg0: !fir.box<!fir.array<?xi32>>)

// CHECK-LABEL: llvm.func @_QPtest_dt_slice
// CHECK:         %[[ALLOCA_SIZE:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:         %[[ALLOCA:.*]] = llvm.alloca %[[ALLOCA_SIZE]] x !llvm.struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr<struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>>
// CHECK:         %[[C20:.*]] = llvm.mlir.constant(20 : index) : i64
// CHECK:         %[[C1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[C10:.*]] = llvm.mlir.constant(10 : i64) : i64
// CHECK:         %[[C2:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK:         %[[ALLOCA_SIZE_V:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[V:.*]] = llvm.alloca %[[ALLOCA_SIZE_V]] x i32 {bindc_name = "v", in_type = i32, operand_segment_sizes = dense<0> : vector<2xi32>, uniq_name = "_QFtest_dt_sliceEv"} : (i64) -> !llvm.ptr<i32>
// CHECK:         %[[ALLOCA_SIZE_X:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[X:.*]] = llvm.alloca %[[ALLOCA_SIZE_X]] x !llvm.array<20 x struct<"_QFtest_dt_sliceTt", (i32, i32)>> {bindc_name = "x", in_type = !fir.array<20x!fir.type<_QFtest_dt_sliceTt{i:i32,j:i32}>>, operand_segment_sizes = dense<0> : vector<2xi32>, uniq_name = "_QFtest_dt_sliceEx"} : (i64) -> !llvm.ptr<array<20 x struct<"_QFtest_dt_sliceTt", (i32, i32)>>>
// CHECK:         %[[BOX0:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>
// CHECK:         %[[ELEM_LEN:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK:         %[[TYPE_CODE:.*]] = llvm.mlir.constant(9 : i32) : i32
// CHECK:         %[[ELEM_LEN_I64:.*]] = llvm.sext %[[ELEM_LEN]] : i32 to i64
// CHECK:         %[[BOX1:.*]] = llvm.insertvalue %[[ELEM_LEN_I64]], %[[BOX0]][1 : i32] : !llvm.struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>
// CHECK:         %[[VERSION:.*]] = llvm.mlir.constant(20180515 : i32) : i32
// CHECK:         %[[BOX2:.*]] = llvm.insertvalue %[[VERSION]], %[[BOX1]][2 : i32] : !llvm.struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>
// CHECK:         %[[RANK:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:         %[[RANK_I8:.*]] = llvm.trunc %[[RANK]] : i32 to i8
// CHECK:         %[[BOX3:.*]] = llvm.insertvalue %[[RANK_I8]], %[[BOX2]][3 : i32] : !llvm.struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>
// CHECK:         %[[TYPE_CODE_I8:.*]] = llvm.trunc %[[TYPE_CODE]] : i32 to i8
// CHECK:         %[[BOX4:.*]] = llvm.insertvalue %[[TYPE_CODE_I8]], %[[BOX3]][4 : i32] : !llvm.struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>
// CHECK:         %[[ATTR:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[ATTR_I8:.*]] = llvm.trunc %[[ATTR]] : i32 to i8
// CHECK:         %[[BOX5:.*]] = llvm.insertvalue %[[ATTR_I8]], %[[BOX4]][5 : i32] : !llvm.struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>
// CHECK:         %[[F18ADDENDUM:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[F18ADDENDUM_I8:.*]] = llvm.trunc %[[F18ADDENDUM]] : i32 to i8
// CHECK:         %[[BOX6:.*]] = llvm.insertvalue %[[F18ADDENDUM_I8]], %[[BOX5]][6 : i32] : !llvm.struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>
// CHECK:         %[[ZERO:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[ELEM_LEN_I64:.*]] = llvm.sext %[[ELEM_LEN]] : i32 to i64    
// CHECK:         %[[ELE_TYPE:.*]] = llvm.mlir.null : !llvm.ptr<struct<"_QFtest_dt_sliceTt", (i32, i32)>>
// CHECK:         %[[C1_0:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[GEP_DTYPE_SIZE:.*]] = llvm.getelementptr %[[ELE_TYPE]][%[[C1_0]]] : (!llvm.ptr<struct<"_QFtest_dt_sliceTt", (i32, i32)>>, i64) -> !llvm.ptr<struct<"_QFtest_dt_sliceTt", (i32, i32)>>
// CHECK:         %[[PTRTOINT_DTYPE_SIZE:.*]] = llvm.ptrtoint %[[GEP_DTYPE_SIZE]] : !llvm.ptr<struct<"_QFtest_dt_sliceTt", (i32, i32)>> to i64
// CHECK:         %[[ADJUSTED_OFFSET:.*]] = llvm.sub %3, %30  : i64
// CHECK:         %[[EXT_SUB:.*]] = llvm.sub %[[C10]], %[[C1]]  : i64
// CHECK:         %[[EXT_ADD:.*]] = llvm.add %[[EXT_SUB]], %[[C2]]  : i64
// CHECK:         %[[EXT_SDIV:.*]] = llvm.sdiv %[[EXT_ADD]], %[[C2]]  : i64
// CHECK:         %[[EXT_ICMP:.*]] = llvm.icmp "sgt" %[[EXT_SDIV]], %[[ZERO]] : i64
// CHECK:         %[[EXT_SELECT:.*]] = llvm.select %[[EXT_ICMP]], %[[EXT_SDIV]], %[[ZERO]] : i1, i64
// CHECK:         %[[BOX7:.*]] = llvm.insertvalue %[[ONE]], %[[BOX6]][7 : i32, 0 : i32, 0 : i32] : !llvm.struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>
// CHECK:         %[[BOX8:.*]] = llvm.insertvalue %[[EXT_SELECT]], %[[BOX7]][7 : i32, 0 : i32, 1 : i32] : !llvm.struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>
// CHECK:         %[[STRIDE_MUL:.*]] = llvm.mul %[[PTRTOINT_DTYPE_SIZE]], %[[C2]]  : i64
// CHECK:         %[[BOX9:.*]] = llvm.insertvalue %[[STRIDE_MUL]], %[[BOX8]][7 : i32, 0 : i32, 2 : i32] : !llvm.struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>
// CHECK:         %[[BASE_PTR:.*]] = llvm.getelementptr %[[X]][%[[ZERO]], %[[ADJUSTED_OFFSET]], 0] : (!llvm.ptr<array<20 x struct<"_QFtest_dt_sliceTt", (i32, i32)>>>, i64, i64) -> !llvm.ptr<array<20 x struct<"_QFtest_dt_sliceTt", (i32, i32)>>>
// CHECK:         %[[ADDR_BITCAST:.*]] = llvm.bitcast %[[BASE_PTR]] : !llvm.ptr<array<20 x struct<"_QFtest_dt_sliceTt", (i32, i32)>>> to !llvm.ptr<i32>
// CHECK:         %[[BOX10:.*]] = llvm.insertvalue %[[ADDR_BITCAST]], %[[BOX9]][0 : i32] : !llvm.struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>
// CHECK:         llvm.store %[[BOX10]], %[[ALLOCA]] : !llvm.ptr<struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>>
// CHECK:         llvm.call @_QPtest_dt_callee(%1) : (!llvm.ptr<struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>>) -> ()

// -----

// Test `fircg.ext_array_coor` conversion.

// Conversion with only shape and indice.

func.func @ext_array_coor0(%arg0: !fir.ref<!fir.array<?xi32>>) {
  %c0 = arith.constant 0 : i64
  %1 = fircg.ext_array_coor %arg0(%c0) <%c0> : (!fir.ref<!fir.array<?xi32>>, i64, i64) -> !fir.ref<i32>
  return
}

// CHECK-LABEL: llvm.func @ext_array_coor0(
// CHECK-SAME:                             %[[ARG0:.*]]: !llvm.ptr<i32>)
// CHECK:         %[[C0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[C1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[C0_1:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[IDX:.*]] = llvm.sub %[[C0]], %[[C1]] : i64
// CHECK:         %[[DIFF0:.*]] = llvm.mul %[[IDX]], %[[C1]] : i64
// CHECK:         %[[SC:.*]] = llvm.mul %[[DIFF0]], %[[C1]]  : i64
// CHECK:         %[[OFFSET:.*]] = llvm.add %[[SC]], %[[C0_1]]  : i64
// CHECK:         %[[BITCAST:.*]] = llvm.bitcast %[[ARG0]] : !llvm.ptr<i32> to !llvm.ptr<i32>
// CHECK:         %{{.*}} = llvm.getelementptr %[[BITCAST]][%[[OFFSET]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>

// Conversion with shift and slice.

func.func @ext_array_coor1(%arg0: !fir.ref<!fir.array<?xi32>>) {
  %c0 = arith.constant 0 : i64
  %1 = fircg.ext_array_coor %arg0(%c0) origin %c0[%c0, %c0, %c0]<%c0> : (!fir.ref<!fir.array<?xi32>>, i64, i64, i64, i64, i64, i64) -> !fir.ref<i32>
  return
}

// CHECK-LABEL: llvm.func @ext_array_coor1(
// CHECK-SAME:                             %[[ARG0:.*]]: !llvm.ptr<i32>)
// CHECK:         %[[C0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[C1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[C0_1:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[IDX:.*]] = llvm.sub %[[C0]], %[[C0]] : i64
// CHECK:         %[[DIFF0:.*]] = llvm.mul %[[IDX]], %[[C0]] : i64
// CHECK:         %[[ADJ:.*]] = llvm.sub %[[C0]], %[[C0]]  : i64
// CHECK:         %[[DIFF1:.*]] = llvm.add %[[DIFF0]], %[[ADJ]]  : i64
// CHECK:         %[[STRIDE:.*]] = llvm.mul %[[DIFF1]], %[[C1]]  : i64
// CHECK:         %[[OFFSET:.*]] = llvm.add %[[STRIDE]], %[[C0_1]]  : i64
// CHECK:         %[[BITCAST:.*]] = llvm.bitcast %[[ARG0]] : !llvm.ptr<i32> to !llvm.ptr<i32>
// CHECK:         %{{.*}} = llvm.getelementptr %[[BITCAST]][%[[OFFSET]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>

// Conversion for a dynamic length char.

func.func @ext_array_coor2(%arg0: !fir.ref<!fir.array<?x!fir.char<1,?>>>) {
  %c0 = arith.constant 0 : i64
  %1 = fircg.ext_array_coor %arg0(%c0) <%c0> : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, i64, i64) -> !fir.ref<i32>
  return
}

// CHECK-LABEL: llvm.func @ext_array_coor2(
// CHECK-SAME:                             %[[ARG0:.*]]: !llvm.ptr<i8>)
// CHECK:         %[[C0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[C1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[C0_1:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[IDX:.*]] = llvm.sub %[[C0]], %[[C1]] : i64
// CHECK:         %[[DIFF0:.*]] = llvm.mul %[[IDX]], %[[C1]] : i64
// CHECK:         %[[SC:.*]] = llvm.mul %[[DIFF0]], %[[C1]]  : i64
// CHECK:         %[[OFFSET:.*]] = llvm.add %[[SC]], %[[C0_1]]  : i64
// CHECK:         %[[BITCAST:.*]] = llvm.bitcast %[[ARG0]] : !llvm.ptr<i8> to !llvm.ptr<i32>
// CHECK:         %{{.*}} = llvm.getelementptr %[[BITCAST]][%[[OFFSET]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>

// Conversion for a `fir.box`.

func.func @ext_array_coor3(%arg0: !fir.box<!fir.array<?xi32>>) {
  %c0 = arith.constant 0 : i64
  %1 = fircg.ext_array_coor %arg0(%c0) <%c0> : (!fir.box<!fir.array<?xi32>>, i64, i64) -> !fir.ref<i32>
  return
}

// CHECK-LABEL: llvm.func @ext_array_coor3(
// CHECK-SAME:                             %[[ARG0:.*]]: !llvm.ptr<struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>>) {
// CHECK:         %[[C0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[C1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[C0_1:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[IDX:.*]] = llvm.sub %[[C0]], %[[C1]] : i64
// CHECK:         %[[DIFF0:.*]] = llvm.mul %[[IDX]], %[[C1]] : i64
// CHECK:         %[[C0_2:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[DIMOFFSET:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[STRIDPOS:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:         %[[GEPSTRIDE:.*]] = llvm.getelementptr %[[ARG0]][%[[C0_2]], 7, %[[DIMOFFSET]], %[[STRIDPOS]]] : (!llvm.ptr<struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>>, i32, i64, i32) -> !llvm.ptr<i64>
// CHECK:         %[[LOADEDSTRIDE:.*]] = llvm.load %[[GEPSTRIDE]] : !llvm.ptr<i64>
// CHECK:         %[[SC:.*]] = llvm.mul %[[DIFF0]], %[[LOADEDSTRIDE]]  : i64
// CHECK:         %[[OFFSET:.*]] = llvm.add %[[SC]], %[[C0_1]]  : i64
// CHECK:         %[[C0_3:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[GEPADDR:.*]] = llvm.getelementptr %[[ARG0]][%[[C0_3]], 0] : (!llvm.ptr<struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>>, i32) -> !llvm.ptr<ptr<i32>>
// CHECK:         %[[LOADEDADDR:.*]] = llvm.load %[[GEPADDR]] : !llvm.ptr<ptr<i32>>
// CHECK:         %[[LOADEDADDRBITCAST:.*]] = llvm.bitcast %[[LOADEDADDR]] : !llvm.ptr<i32> to !llvm.ptr<i8>
// CHECK:         %[[GEPADDROFFSET:.*]] = llvm.getelementptr %[[LOADEDADDRBITCAST]][%[[OFFSET]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
// CHECK:         %{{.*}} = llvm.bitcast %[[GEPADDROFFSET]] : !llvm.ptr<i8> to !llvm.ptr<i32>

// Conversion with non zero shift and slice.

func.func @ext_array_coor4(%arg0: !fir.ref<!fir.array<100xi32>>) {
  %c0 = arith.constant 0 : i64
  %c10 = arith.constant 10 : i64
  %c20 = arith.constant 20 : i64
  %c1 = arith.constant 1 : i64
  %1 = fircg.ext_array_coor %arg0(%c0) origin %c0[%c10, %c20, %c1]<%c1> : (!fir.ref<!fir.array<100xi32>>, i64, i64, i64, i64, i64, i64) -> !fir.ref<i32>
  return
}

// CHECK-LABEL: llvm.func @ext_array_coor4(
// CHECK:                                  %[[ARG0:.*]]: !llvm.ptr<array<100 x i32>>) {
// CHECK:         %[[C0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[C10:.*]] = llvm.mlir.constant(10 : i64) : i64
// CHECK:         %[[C20:.*]] = llvm.mlir.constant(20 : i64) : i64
// CHECK:         %[[C1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[C1_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[C0_1:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[IDX:.*]] = llvm.sub %[[C1]], %[[C0]]  : i64
// CHECK:         %[[DIFF0:.*]] = llvm.mul %[[IDX]], %[[C1]]  : i64
// CHECK:         %[[ADJ:.*]] = llvm.sub %[[C10]], %[[C0]]  : i64
// CHECK:         %[[DIFF1:.*]] = llvm.add %[[DIFF0]], %[[ADJ]]  : i64
// CHECK:         %[[STRIDE:.*]] = llvm.mul %[[DIFF1]], %[[C1_1]] : i64
// CHECK:         %[[OFFSET:.*]] = llvm.add %[[STRIDE]], %[[C0_1]]  : i64
// CHECK:         %[[BITCAST:.*]] = llvm.bitcast %[[ARG0]] : !llvm.ptr<array<100 x i32>> to !llvm.ptr<i32>
// CHECK:         %{{.*}} = llvm.getelementptr %[[BITCAST]][%[[OFFSET]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>

// -----

// Check `fircg.ext_rebox` conversion to LLVM IR dialect

// Test applying slice on fir.box. Note that the slice is 1D where as the array is 2D.
//   subroutine foo(x)
//     real :: x(3:, 4:)
//     call bar(x(5, 6:80:3))
//   end subroutine

func.func private @bar1(!fir.box<!fir.array<?xf32>>)
func.func @test_rebox_1(%arg0: !fir.box<!fir.array<?x?xf32>>) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c80 = arith.constant 80 : index
  %0 = fir.undefined index
  %3 = fircg.ext_rebox %arg0 origin %c3, %c4[%c5, %0, %0, %c6, %c80, %c3] : (!fir.box<!fir.array<?x?xf32>>, index, index, index, index, index, index, index, index) -> !fir.box<!fir.array<?xf32>>
  fir.call @bar1(%3) : (!fir.box<!fir.array<?xf32>>) -> ()
  return
}
//CHECK-LABEL:  llvm.func @bar1
//CHECK-LABEL:  llvm.func @test_rebox_1
//CHECK-SAME:   %[[ARG0:.*]]: !llvm.ptr<struct<(ptr<f32>, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>>
//CHECK:    %[[ONE_1:.*]] = llvm.mlir.constant(1 : i32) : i32
//CHECK:    %[[RESULT_BOX_REF:.*]] = llvm.alloca %[[ONE_1]] x !llvm.struct<(ptr<f32>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr<struct<(ptr<f32>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>>
//CHECK:    %[[THREE:.*]] = llvm.mlir.constant(3 : index) : i64
//CHECK:    %[[FOUR:.*]] = llvm.mlir.constant(4 : index) : i64
//CHECK:    %[[FIVE:.*]] = llvm.mlir.constant(5 : index) : i64
//CHECK:    %[[SIX:.*]] = llvm.mlir.constant(6 : index) : i64
//CHECK:    %[[EIGHTY:.*]] = llvm.mlir.constant(80 : index) : i64
//CHECK:    %[[RBOX:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
//CHECK:    %[[ELEM_SIZE:.*]] = llvm.mlir.constant(4 : i32) : i32
//CHECK:    %[[FLOAT_TYPE:.*]] = llvm.mlir.constant(27 : i32) : i32
//CHECK:    %[[ELEM_SIZE_I64:.*]] = llvm.sext %[[ELEM_SIZE]] : i32 to i64
//CHECK:    %[[RBOX_TMP1:.*]] = llvm.insertvalue %[[ELEM_SIZE_I64]], %[[RBOX]][1 : i32] : !llvm.struct<(ptr<f32>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
//CHECK:    %[[CFI_VERSION:.*]] = llvm.mlir.constant(20180515 : i32) : i32
//CHECK:    %[[RBOX_TMP2:.*]] = llvm.insertvalue %[[CFI_VERSION]], %[[RBOX_TMP1]][2 : i32] : !llvm.struct<(ptr<f32>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
//CHECK:    %[[RANK:.*]] = llvm.mlir.constant(1 : i32) : i32
//CHECK:    %[[RANK_I8:.*]] = llvm.trunc %[[RANK]] : i32 to i8
//CHECK:    %[[RBOX_TMP3:.*]] = llvm.insertvalue %[[RANK_I8]], %[[RBOX_TMP2]][3 : i32] : !llvm.struct<(ptr<f32>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
//CHECK:    %[[FLOAT_TYPE_I8:.*]] = llvm.trunc %[[FLOAT_TYPE]] : i32 to i8
//CHECK:    %[[RBOX_TMP4:.*]] = llvm.insertvalue %[[FLOAT_TYPE_I8]], %[[RBOX_TMP3]][4 : i32] : !llvm.struct<(ptr<f32>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
//CHECK:    %[[OTHER_ATTR:.*]] = llvm.mlir.constant(0 : i32) : i32
//CHECK:    %[[OTHER_ATTR_I8:.*]] = llvm.trunc %[[OTHER_ATTR]] : i32 to i8
//CHECK:    %[[RBOX_TMP5:.*]] = llvm.insertvalue %[[OTHER_ATTR_I8]], %[[RBOX_TMP4]][5 : i32] : !llvm.struct<(ptr<f32>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
//CHECK:    %[[ADDENDUM:.*]] = llvm.mlir.constant(0 : i32) : i32
//CHECK:    %[[ADDENDUM_I8:.*]] = llvm.trunc %[[ADDENDUM]] : i32 to i8
//CHECK:    %[[RBOX_TMP6:.*]] = llvm.insertvalue %[[ADDENDUM_I8]], %[[RBOX_TMP5]][6 : i32] : !llvm.struct<(ptr<f32>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
//CHECK:    %[[DIM1:.*]] = llvm.mlir.constant(0 : i64) : i64
//CHECK:    %[[GEP_ZERO_1:.*]] = llvm.mlir.constant(0 : i32) : i32
//CHECK:    %[[LB1_IDX:.*]] = llvm.mlir.constant(2 : i32) : i32
//CHECK:    %[[DIM1_STRIDE_REF:.*]] = llvm.getelementptr %[[ARG0]][%[[GEP_ZERO_1]], 7, %[[DIM1]], %[[LB1_IDX]]] : (!llvm.ptr<struct<(ptr<f32>, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>>, i32, i64, i32) -> !llvm.ptr<i64>
//CHECK:    %[[DIM1_STRIDE:.*]] = llvm.load %[[DIM1_STRIDE_REF]] : !llvm.ptr<i64>
//CHECK:    %[[DIM2:.*]] = llvm.mlir.constant(1 : i64) : i64
//CHECK:    %[[GEP_ZERO_2:.*]] = llvm.mlir.constant(0 : i32) : i32
//CHECK:    %[[STRIDE2_IDX:.*]] = llvm.mlir.constant(2 : i32) : i32
//CHECK:    %[[DIM2_STRIDE_REF:.*]] = llvm.getelementptr %[[ARG0]][%[[GEP_ZERO_2]], 7, %[[DIM2]], %[[STRIDE2_IDX]]] : (!llvm.ptr<struct<(ptr<f32>, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>>, i32, i64, i32) -> !llvm.ptr<i64>
//CHECK:    %[[DIM2_STRIDE:.*]] = llvm.load %[[DIM2_STRIDE_REF]] : !llvm.ptr<i64>
//CHECK:    %[[ZERO_1:.*]] = llvm.mlir.constant(0 : i32) : i32
//CHECK:    %[[SOURCE_ARRAY_PTR:.*]] = llvm.getelementptr %[[ARG0]][%[[ZERO_1]], 0] : (!llvm.ptr<struct<(ptr<f32>, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>>, i32) -> !llvm.ptr<ptr<f32>>
//CHECK:    %[[SOURCE_ARRAY:.*]] = llvm.load %[[SOURCE_ARRAY_PTR]] : !llvm.ptr<ptr<f32>>
//CHECK:    %[[ZERO_ELEMS:.*]] = llvm.mlir.constant(0 : i64) : i64
//CHECK:    %[[SOURCE_ARRAY_I8PTR:.*]] = llvm.bitcast %[[SOURCE_ARRAY]] : !llvm.ptr<f32> to !llvm.ptr<i8>
//CHECK:    %[[DIM1_LB_DIFF:.*]] = llvm.sub %[[FIVE]], %[[THREE]]  : i64
//CHECK:    %[[DIM1_LB_OFFSET:.*]] = llvm.mul %[[DIM1_LB_DIFF]], %[[DIM1_STRIDE]]  : i64
//CHECK:    %[[RESULT_PTR_DIM1:.*]] = llvm.getelementptr %[[SOURCE_ARRAY_I8PTR]][%[[DIM1_LB_OFFSET]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
//CHECK:    %[[DIM2_LB_DIFF:.*]] = llvm.sub %[[SIX]], %[[FOUR]]  : i64
//CHECK:    %[[DIM2_LB_OFFSET:.*]] = llvm.mul %[[DIM2_LB_DIFF]], %[[DIM2_STRIDE]]  : i64
//CHECK:    %[[RESULT_PTR_I8:.*]] = llvm.getelementptr %[[RESULT_PTR_DIM1]][%[[DIM2_LB_OFFSET]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
//CHECK:    %[[RESULT_UB_LB_DIFF:.*]] = llvm.sub %[[EIGHTY]], %[[SIX]]  : i64
//CHECK:    %[[RESULT_UB_LB_DIFF_PLUS_STRIDE:.*]] = llvm.add %[[RESULT_UB_LB_DIFF]], %[[THREE]]  : i64
//CHECK:    %[[RESULT_NELEMS_TMP:.*]] = llvm.sdiv %[[RESULT_UB_LB_DIFF_PLUS_STRIDE]], %[[THREE]]  : i64
//CHECK:    %[[RESULT_IF_NON_ZERO:.*]] = llvm.icmp "sgt" %[[RESULT_NELEMS_TMP]], %[[ZERO_ELEMS]] : i64
//CHECK:    %[[RESULT_NELEMS:.*]] = llvm.select %[[RESULT_IF_NON_ZERO]], %[[RESULT_NELEMS_TMP]], %[[ZERO_ELEMS]] : i1, i64
//CHECK:    %[[RESULT_STRIDE:.*]] = llvm.mul %[[THREE]], %[[DIM2_STRIDE]]  : i64
//CHECK:    %[[RESULT_LB:.*]] = llvm.mlir.constant(1 : i64) : i64
//CHECK:    %[[RBOX_TMP7_1:.*]] = llvm.insertvalue %[[RESULT_LB]], %[[RBOX_TMP6]][7 : i32, 0 : i32, 0 : i32] : !llvm.struct<(ptr<f32>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
//CHECK:    %[[RBOX_TMP7_2:.*]] = llvm.insertvalue %[[RESULT_NELEMS]], %[[RBOX_TMP7_1]][7 : i32, 0 : i32, 1 : i32] : !llvm.struct<(ptr<f32>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
//CHECK:    %[[RBOX_TMP7_3:.*]] = llvm.insertvalue %[[RESULT_STRIDE]], %[[RBOX_TMP7_2]][7 : i32, 0 : i32, 2 : i32] : !llvm.struct<(ptr<f32>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
//CHECK:    %[[RESULT_PTR_F32:.*]] = llvm.bitcast %[[RESULT_PTR_I8]] : !llvm.ptr<i8> to !llvm.ptr<f32>
//CHECK:    %[[RESULT_BOX:.*]] = llvm.insertvalue %[[RESULT_PTR_F32]], %[[RBOX_TMP7_3]][0 : i32] : !llvm.struct<(ptr<f32>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
//CHECK:    llvm.store %[[RESULT_BOX]], %[[RESULT_BOX_REF]] : !llvm.ptr<struct<(ptr<f32>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>>
//CHECK:    llvm.call @bar1(%[[RESULT_BOX_REF]]) : (!llvm.ptr<struct<(ptr<f32>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>>) -> ()


// Test a rebox of an array section like x(3:60:9)%c(2:8) with both a triplet, a component and a substring where x is a fir.box.
func.func private @bar(!fir.box<!fir.array<?x!fir.char<1,?>>>)
func.func @foo(%arg0: !fir.box<!fir.array<?x!fir.type<t{i:i32,c:!fir.char<1,10>}>>>) {
  %c3_i64 = arith.constant 3 : i64
  %c60_i64 = arith.constant 60 : i64
  %c9_i64 = arith.constant 9 : i64
  %c1_i64 = arith.constant 1 : i64
  %c7_i64 = arith.constant 7 : i64
  %0 = fir.field_index c, !fir.type<t{i:i32,c:!fir.char<1,10>}>
  %1 = fircg.ext_rebox %arg0[%c3_i64, %c60_i64, %c9_i64] path %0 substr %c1_i64, %c7_i64 : (!fir.box<!fir.array<?x!fir.type<t{i:i32,c:!fir.char<1,10>}>>>, i64, i64, i64, !fir.field, i64, i64) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
  fir.call @bar(%1) : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> ()
  return
}

//CHECK: llvm.func @bar(!llvm.ptr<struct<(ptr<i8>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>>) attributes {sym_visibility = "private"}
//CHECK-LABEL: llvm.func @foo
//CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr<struct<(ptr<struct<"t", (i32, array<10 x i8>)>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>>
//CHECK:   %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
//CHECK:   %[[RESULT_BOX_REF:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<(ptr<i8>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr<struct<(ptr<i8>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>>
//CHECK:   %[[RESULT_LB:.*]] = llvm.mlir.constant(3 : i64) : i64
//CHECK:   %[[RESULT_UB:.*]] = llvm.mlir.constant(60 : i64) : i64
//CHECK:   %[[RESULT_STRIDE:.*]] = llvm.mlir.constant(9 : i64) : i64
//CHECK:   %[[COMPONENT_OFFSET_1:.*]] = llvm.mlir.constant(1 : i64) : i64
//CHECK:   %[[ELEM_SIZE:.*]] = llvm.mlir.constant(7 : i64) : i64
//CHECK:   %[[TYPE_CHAR:.*]] = llvm.mlir.constant(40 : i32) : i32
//CHECK:   %[[RBOX_TMP1:.*]] = llvm.insertvalue %[[ELEM_SIZE]], %{{.*}}[1 : i32] : !llvm.struct<(ptr<i8>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
//CHECK:   %[[RBOX_TMP2:.*]] = llvm.insertvalue %{{.*}}, %[[RBOX_TMP1]][2 : i32] : !llvm.struct<(ptr<i8>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
//CHECK:   %[[RANK:.*]] = llvm.mlir.constant(1 : i32) : i32
//CHECK:   %[[RANK_I8:.*]] = llvm.trunc %[[RANK]] : i32 to i8
//CHECK:   %[[RBOX_TMP3:.*]] = llvm.insertvalue %[[RANK_I8]], %[[RBOX_TMP2]][3 : i32] : !llvm.struct<(ptr<i8>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
//CHECK:   %[[TYPE_CHAR_I8:.*]] = llvm.trunc %[[TYPE_CHAR]] : i32 to i8
//CHECK:   %[[RBOX_TMP4:.*]] = llvm.insertvalue %[[TYPE_CHAR_I8]], %[[RBOX_TMP3]][4 : i32] : !llvm.struct<(ptr<i8>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
//CHECK:   %[[RBOX_TMP5:.*]] = llvm.insertvalue %{{.*}}, %[[RBOX_TMP4]][5 : i32] : !llvm.struct<(ptr<i8>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
//CHECK:   %[[ADDENDUM:.*]] = llvm.mlir.constant(0 : i32) : i32
//CHECK:   %[[ADDENDUM_I8:.*]] = llvm.trunc %[[ADDENDUM]] : i32 to i8
//CHECK:   %[[RBOX_TMP6:.*]] = llvm.insertvalue %[[ADDENDUM_I8]], %[[RBOX_TMP5]][6 : i32] : !llvm.struct<(ptr<i8>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
//CHECK:   %[[DIM1:.*]] = llvm.mlir.constant(0 : i64) : i64
//CHECK:   %[[ZERO_3:.*]] = llvm.mlir.constant(0 : i32) : i32
//CHECK:   %[[STRIDE_IDX:.*]] = llvm.mlir.constant(2 : i32) : i32
//CHECK:   %[[SRC_STRIDE_PTR:.*]] = llvm.getelementptr %[[ARG0]][%[[ZERO_3]], 7, %[[DIM1]], %[[STRIDE_IDX]]] : (!llvm.ptr<struct<(ptr<struct<"t", (i32, array<10 x i8>)>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>>, i32, i64, i32) -> !llvm.ptr<i64>
//CHECK:   %[[SRC_STRIDE:.*]] = llvm.load %[[SRC_STRIDE_PTR]] : !llvm.ptr<i64>
//CHECK:   %[[ZERO_4:.*]] = llvm.mlir.constant(0 : i32) : i32
//CHECK:   %[[SRC_ARRAY_PTR:.*]] = llvm.getelementptr %[[ARG0]][%[[ZERO_4]], 0] : (!llvm.ptr<struct<(ptr<struct<"t", (i32, array<10 x i8>)>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>>, i32) -> !llvm.ptr<ptr<struct<"t", (i32, array<10 x i8>)>>>
//CHECK:   %[[SRC_ARRAY:.*]] = llvm.load %[[SRC_ARRAY_PTR]] : !llvm.ptr<ptr<struct<"t", (i32, array<10 x i8>)>>>
//CHECK:   %[[ZERO_6:.*]] = llvm.mlir.constant(0 : i64) : i64
//CHECK:   %[[SRC_CAST:.*]] = llvm.bitcast %[[SRC_ARRAY]] : !llvm.ptr<struct<"t", (i32, array<10 x i8>)>> to !llvm.ptr<struct<"t", (i32, array<10 x i8>)>>
//CHECK:   %[[TMP_COMPONENT:.*]] = llvm.getelementptr %[[SRC_CAST]][%[[ZERO_6]], 1] : (!llvm.ptr<struct<"t", (i32, array<10 x i8>)>>, i64) -> !llvm.ptr<struct<"t", (i32, array<10 x i8>)>>
//CHECK:   %[[COMPONENT:.*]] = llvm.getelementptr %[[TMP_COMPONENT]][%[[COMPONENT_OFFSET_1]]] : (!llvm.ptr<struct<"t", (i32, array<10 x i8>)>>, i64) -> !llvm.ptr<struct<"t", (i32, array<10 x i8>)>>
//CHECK:   %[[COMPONENT_CAST:.*]] = llvm.bitcast %[[COMPONENT]] : !llvm.ptr<struct<"t", (i32, array<10 x i8>)>> to !llvm.ptr<i8>
//CHECK:   %[[SRC_LB:.*]] = llvm.mlir.constant(1 : i64) : i64
//CHECK:   %[[RESULT_TMP0:.*]] = llvm.sub %[[RESULT_LB]], %[[SRC_LB]]  : i64
//CHECK:   %[[RESULT_OFFSET_START:.*]] = llvm.mul %[[RESULT_TMP0]], %[[SRC_STRIDE]]  : i64
//CHECK:   %[[RESULT_PTR_I8:.*]] = llvm.getelementptr %[[COMPONENT_CAST]][%[[RESULT_OFFSET_START]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
//CHECK:   %[[RESULT_TMP1:.*]] = llvm.sub %[[RESULT_UB]], %[[RESULT_LB]]  : i64
//CHECK:   %[[RESULT_TMP2:.*]] = llvm.add %[[RESULT_TMP1]], %[[RESULT_STRIDE]]  : i64
//CHECK:   %[[RESULT_TMP3:.*]] = llvm.sdiv %[[RESULT_TMP2]], %[[RESULT_STRIDE]]  : i64
//CHECK:   %[[RESULT_TMP_PRED:.*]] = llvm.icmp "sgt" %[[RESULT_TMP3]], %[[ZERO_6]] : i64
//CHECK:   %[[RESULT_NELEMS:.*]] = llvm.select %[[RESULT_TMP_PRED]], %[[RESULT_TMP3]], %[[ZERO_6]] : i1, i64
//CHECK:   %[[RESULT_TOTAL_STRIDE:.*]] = llvm.mul %[[RESULT_STRIDE]], %[[SRC_STRIDE]]  : i64
//CHECK:   %[[RESULT_LB:.*]] = llvm.mlir.constant(1 : i64) : i64
//CHECK:   %[[RBOX_TMP7_1:.*]] = llvm.insertvalue %[[RESULT_LB]], %[[RBOX_TMP6]][7 : i32, 0 : i32, 0 : i32] : !llvm.struct<(ptr<i8>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
//CHECK:   %[[RBOX_TMP7_2:.*]] = llvm.insertvalue %[[RESULT_NELEMS]], %[[RBOX_TMP7_1]][7 : i32, 0 : i32, 1 : i32] : !llvm.struct<(ptr<i8>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
//CHECK:   %[[RBOX_TMP7_3:.*]] = llvm.insertvalue %[[RESULT_TOTAL_STRIDE]], %[[RBOX_TMP7_2]][7 : i32, 0 : i32, 2 : i32] : !llvm.struct<(ptr<i8>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
//CHECK:   %[[RESULT_PTR_CAST:.*]] = llvm.bitcast %[[RESULT_PTR_I8]] : !llvm.ptr<i8> to !llvm.ptr<i8>
//CHECK:   %[[RESULT_BOX:.*]] = llvm.insertvalue %[[RESULT_PTR_CAST]], %[[RBOX_TMP7_3]][0 : i32] : !llvm.struct<(ptr<i8>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
//CHECK:   llvm.store %[[RESULT_BOX]], %[[RESULT_BOX_REF]] : !llvm.ptr<struct<(ptr<i8>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>>
//CHECK:   llvm.call @bar(%[[RESULT_BOX_REF]]) : (!llvm.ptr<struct<(ptr<i8>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>>) -> ()
//CHECK:   llvm.return
//CHECK: }

// -----

// Test `fir.coordinate_of` conversion (items inside `!fir.box`)

// 1. COMPLEX TYPE (`fir.complex` is a special case)
// Complex type wrapped in `fir.ref`
func.func @coordinate_ref_complex(%arg0: !fir.ref<!fir.complex<16>>) {
  %arg1 = llvm.mlir.constant(0 : i32) : i32
  %p = fir.coordinate_of %arg0, %arg1 : (!fir.ref<!fir.complex<16>>, i32) -> !fir.ref<f32>
  return
}
// CHECK-LABEL: llvm.func @coordinate_ref_complex
// CHECK-SAME:  %[[ARG0:.*]]: !llvm.ptr<struct<(f128, f128)>>
// CHECK:    %[[C0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:    %{{.*}} = llvm.getelementptr %[[ARG0]][%[[C0]], 0] : (!llvm.ptr<struct<(f128, f128)>>, i64) -> !llvm.ptr<f32>
// CHECK-NEXT:    llvm.return

// -----

// Complex type wrapped in `fir.box`
func.func @coordinate_box_complex(%arg0: !fir.box<!fir.complex<16>>) {
  %arg1 = llvm.mlir.constant(0 : i32) : i32
  %p = fir.coordinate_of %arg0, %arg1 : (!fir.box<!fir.complex<16>>, i32) -> !fir.ref<f32>
  return
}
// CHECK-LABEL: llvm.func @coordinate_box_complex
// CHECK-SAME:  %[[BOX:.*]]: !llvm.ptr<struct<(ptr<struct<(f128, f128)>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>
// CHECK:    %[[C0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:    %{{.*}} = llvm.getelementptr %[[BOX]][%[[C0]], 0] : (!llvm.ptr<struct<(ptr<struct<(f128, f128)>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>, i64) -> !llvm.ptr<f32>
// CHECK-NEXT:    llvm.return

// -----

// Test `fir.coordinate_of` conversion (items inside `!fir.box`)

// 2. BOX TYPE (objects wrapped in `fir.box`)
// Derived type - basic case (1 index)
func.func @coordinate_box_derived_1(%arg0: !fir.box<!fir.type<derived_1{field_1:i32, field_2:i32}>>) {
  %idx = fir.field_index field_2, !fir.type<derived_1{field_1:i32, field_2:i32}>
  %q = fir.coordinate_of %arg0, %idx : (!fir.box<!fir.type<derived_1{field_1:i32, field_2:i32}>>, !fir.field) -> !fir.ref<i32>
  return
}
// CHECK-LABEL: llvm.func @coordinate_box_derived_1
// CHECK-SAME: %[[BOX:.*]]: !llvm.ptr<struct<(ptr<struct<"derived_1", (i32, i32)>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, ptr<i8>, array<1 x i64>)>>)
// CHECK:    %[[COORDINATE:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:    %[[C0_3:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:    %[[C0_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:    %[[DERIVED_ADDR:.*]] = llvm.getelementptr %[[BOX]][%[[C0_1]], 0] : (!llvm.ptr<struct<(ptr<struct<"derived_1", (i32, i32)>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, ptr<i8>, array<1 x i64>)>>, i32) -> !llvm.ptr<ptr<struct<"derived_1", (i32, i32)>>>
// CHECK:    %[[DERIVED_VAL:.*]] = llvm.load %[[DERIVED_ADDR]] : !llvm.ptr<ptr<struct<"derived_1", (i32, i32)>>>
// CHECK:    %[[DERIVED_CAST:.*]] = llvm.bitcast %[[DERIVED_VAL]] : !llvm.ptr<struct<"derived_1", (i32, i32)>> to !llvm.ptr<struct<"derived_1", (i32, i32)>>
// CHECK:    %[[SUBOBJECT_ADDR:.*]] = llvm.getelementptr %[[DERIVED_CAST]][%[[C0_3]], 1] : (!llvm.ptr<struct<"derived_1", (i32, i32)>>, i64) -> !llvm.ptr<i32>
// CHECK:    %[[CAST_TO_I8_PTR:.*]] = llvm.bitcast %7 : !llvm.ptr<i32> to !llvm.ptr<i8>
// CHECK:    %{{.*}} = llvm.bitcast %[[CAST_TO_I8_PTR]] : !llvm.ptr<i8> to !llvm.ptr<i32>
// CHECK-NEXT:    llvm.return

// Derived type - basic case (2 indices)
func.func @coordinate_box_derived_2(%arg0: !fir.box<!fir.type<derived_2{field_1:!fir.type<another_derived{inner1:i32, inner2:f32}>, field_2:i32}>>) {
  %idx0 = fir.field_index field_1, !fir.type<derived_2{field_1:!fir.type<another_derived{inner1:i32, inner2:f32}>, field_2:i32}>
  %idx1 = fir.field_index inner2, !fir.type<another_derived{inner1:i32, inner2:f32}>
  %q = fir.coordinate_of %arg0, %idx0, %idx1 : (!fir.box<!fir.type<derived_2{field_1:!fir.type<another_derived{inner1:i32, inner2:f32}>, field_2:i32}>>, !fir.field, !fir.field) -> !fir.ref<i32>
  return
}

// CHECK-LABEL: llvm.func @coordinate_box_derived_2
// CHECK-SAME: (%[[BOX:.*]]: !llvm.ptr<struct<(ptr<struct<"derived_2", (struct<"another_derived", (i32, f32)>, i32)>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, ptr<i8>, array<1 x i64>)>>)
// CHECK-NEXT:    %[[C0_0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[C0_3:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[C0_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[DERIVED_ADDR:.*]] = llvm.getelementptr %[[BOX]][%[[C0_1]], 0] : (!llvm.ptr<struct<(ptr<struct<"derived_2", (struct<"another_derived", (i32, f32)>, i32)>>, i{{.*}}, i{{.*}}32, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, ptr<i8>, array<1 x i64>)>>, i32) -> !llvm.ptr<ptr<struct<"derived_2", (struct<"another_derived", (i32, f32)>, i32)>>>
// CHECK-NEXT:    %[[DERIVED_VAL:.*]] = llvm.load %[[DERIVED_ADDR]] : !llvm.ptr<ptr<struct<"derived_2", (struct<"another_derived", (i32, f32)>, i32)>>>
// CHECK-NEXT:    %[[DERIVED_CAST_I8_PTR:.*]] = llvm.bitcast %[[DERIVED_VAL]] : !llvm.ptr<struct<"derived_2", (struct<"another_derived", (i32, f32)>, i32)>> to !llvm.ptr<struct<"derived_2", (struct<"another_derived", (i32, f32)>, i32)>>
// CHECK-NEXT:    %[[ANOTHER_DERIVED_ADDR:.*]] = llvm.getelementptr %[[DERIVED_CAST_I8_PTR]][%[[C0_3]], 0] : (!llvm.ptr<struct<"derived_2", (struct<"another_derived", (i32, f32)>, i32)>>, i64) -> !llvm.ptr<struct<"another_derived", (i32, f32)>>
// CHECK-NEXT:    %[[ANOTHER_DERIVED_ADDR_AS_VOID_PTR:.*]] = llvm.bitcast %[[ANOTHER_DERIVED_ADDR]] : !llvm.ptr<struct<"another_derived", (i32, f32)>> to !llvm.ptr<i8>
// CHECK-NEXT:    %[[ANOTHER_DERIVED_RECAST:.*]] = llvm.bitcast %[[ANOTHER_DERIVED_ADDR_AS_VOID_PTR]] : !llvm.ptr<i8> to !llvm.ptr<struct<"another_derived", (i32, f32)>>
// CHECK-NEXT:    %[[SUBOBJECT_ADDR:.*]] = llvm.getelementptr %[[ANOTHER_DERIVED_RECAST]][%[[C0_3]], 1] : (!llvm.ptr<struct<"another_derived", (i32, f32)>>, i64) -> !llvm.ptr<f32>
// CHECK-NEXT:    %[[SUBOBJECT_AS_VOID_PTR:.*]] = llvm.bitcast %[[SUBOBJECT_ADDR]] : !llvm.ptr<f32> to !llvm.ptr<i8>
// CHECK-NEXT:    %{{.*}} = llvm.bitcast %[[SUBOBJECT_AS_VOID_PTR]] : !llvm.ptr<i8> to !llvm.ptr<i32>
// CHECK-NEXT:   llvm.return

// TODO: Derived type - special case with `fir.len_param_index`

// -----

// Test `fir.coordinate_of` conversion (items inside `!fir.box`)

// 3. BOX TYPE - `fir.array` wrapped in `fir.box`
// `fir.array` inside a `fir.box` (1d)
func.func @coordinate_box_array_1d(%arg0: !fir.box<!fir.array<10 x f32>>, %arg1: index) {
  %p = fir.coordinate_of %arg0, %arg1 : (!fir.box<!fir.array<10 x f32>>, index) -> !fir.ref<f32>
  return
}
// CHECK-LABEL: llvm.func @coordinate_box_array_1d
// CHECK-SAME:  %[[BOX:.*]]: !llvm.ptr<struct<(ptr<array<10 x f32>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>>
// CHECK-SAME:  %[[COORDINATE:.*]]: i64
// CHECK-NEXT:  %{{.*}} = llvm.mlir.constant(0 : i64) : i64
// There's only one box here. Its index is `0`. Generate it.
// CHECK-NEXT:  %[[BOX_IDX:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:       %[[ARRAY_ADDR:.*]] = llvm.getelementptr %[[BOX]][%[[BOX_IDX]], 0] : (!llvm.ptr<struct<(ptr<array<10 x f32>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>>, i32) -> !llvm.ptr<ptr<array<10 x f32>>>
// CHECK-NEXT:  %[[ARRAY_OBJECT:.*]] = llvm.load %[[ARRAY_ADDR]] : !llvm.ptr<ptr<array<10 x f32>>>
// CHECK-NEXT:  %[[OFFSET_INIT:.*]] = llvm.mlir.constant(0 : i64) : i64
// Same as [[BOX_IDX]], just recreated.
// CHECK-NEXT:  %[[BOX_IDX_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// Index of the array that contains the CFI_dim_t objects
// CHECK-NEXT:  %[[CFI_DIM_IDX:.*]] = llvm.mlir.constant(7 : i32) : i32
// Index of the 1st CFI_dim_t object (corresonds the the 1st dimension)
// CHECK-NEXT:  %[[DIM_1_IDX:.*]] = llvm.mlir.constant(0 : i64) : i64
// Index of the memory stride within a CFI_dim_t object
// CHECK-NEXT:  %[[DIM_1_MEM_STRIDE:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:  %[[DIM_1_MEM_STRIDE_ADDR:.*]] = llvm.getelementptr %[[BOX]][%[[BOX_IDX_1]], 7, %[[DIM_1_IDX]], %[[DIM_1_MEM_STRIDE]]] : (!llvm.ptr<struct<(ptr<array<10 x f32>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>>, i32, i64, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:  %[[DIM_1_MEM_STRIDE_VAL:.*]] = llvm.load %[[DIM_1_MEM_STRIDE_ADDR]] : !llvm.ptr<i64>
// CHECK-NEXT:  %[[BYTE_OFFSET:.*]] = llvm.mul %[[COORDINATE]], %[[DIM_1_MEM_STRIDE_VAL]]  : i64
// CHECK-NEXT:  %[[SUBOJECT_OFFSET:.*]] = llvm.add %[[BYTE_OFFSET]], %[[OFFSET_INIT]]  : i64
// CHECK-NEXT:  %[[ARRAY_OBJECT_AS_VOID_PTR:.*]] = llvm.bitcast %[[ARRAY_OBJECT]] : !llvm.ptr<array<10 x f32>> to !llvm.ptr<i8>
// CHECK-NEXT:  %[[SUBOBJECT_ADDR:.*]] = llvm.getelementptr %[[ARRAY_OBJECT_AS_VOID_PTR]][%[[SUBOJECT_OFFSET]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
// CHECK-NEXT:  %[[RETURN_VAL:.*]] = llvm.bitcast %[[SUBOBJECT_ADDR]] : !llvm.ptr<i8> to !llvm.ptr<f32>
// CHECK-NEXT:  llvm.return

// `fir.array` inside a `fir.box` (1d) - dynamic size
func.func @coordinate_of_box_dynamic_array_1d(%arg0: !fir.box<!fir.array<? x f32>>, %arg1: index) {
  %p = fir.coordinate_of %arg0, %arg1 : (!fir.box<!fir.array<? x f32>>, index) -> !fir.ref<f32>
  return
}
// CHECK-LABEL: llvm.func @coordinate_of_box_dynamic_array_1d
// CHECK-SAME:  %[[BOX:.*]]: !llvm.ptr<struct<(ptr<f32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>>
// CHECK-SAME:  %[[COORDINATE:.*]]: i64
// CHECK-NEXT:  %{{.*}} = llvm.mlir.constant(0 : i64) : i64
// There's only one box here. Its index is `0`. Generate it.
// CHECK-NEXT:  %[[BOX_IDX:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:  %[[BOX_1ST_ELEM_IDX:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:  %[[ARRAY_ADDR:.*]] = llvm.getelementptr %[[BOX]][%[[BOX_IDX]], 0] : (!llvm.ptr<struct<(ptr<f32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>>, i32) -> !llvm.ptr<ptr<f32>>
// CHECK-NEXT:  %[[ARRAY_OBJECT:.*]] = llvm.load %[[ARRAY_ADDR]] : !llvm.ptr<ptr<f32>>
// CHECK-NEXT:  %[[OFFSET_INIT:.*]] = llvm.mlir.constant(0 : i64) : i64
// Same as [[BOX_IDX]], just recreated.
// CHECK-NEXT:  %[[BOX_IDX_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// Index of the array that contains the CFI_dim_t objects
// CHECK-NEXT:  %[[CFI_DIM_IDX:.*]] = llvm.mlir.constant(7 : i32) : i32
// Index of the 1st CFI_dim_t object (corresonds the the 1st dimension)
// CHECK-NEXT:  %[[DIM_1_IDX:.*]] = llvm.mlir.constant(0 : i64) : i64
// Index of the memory stride within a CFI_dim_t object
// CHECK-NEXT:  %[[DIM_1_MEM_STRIDE:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:  %[[DIM_1_MEM_STRIDE_ADDR:.*]] = llvm.getelementptr %[[BOX]][%[[BOX_IDX_1]], 7, %[[DIM_1_IDX]], %[[DIM_1_MEM_STRIDE]]] : (!llvm.ptr<struct<(ptr<f32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i64>>)>>, i32, i64, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:  %[[DIM_1_MEM_STRIDE_VAL:.*]] = llvm.load %[[DIM_1_MEM_STRIDE_ADDR]] : !llvm.ptr<i64>
// CHECK-NEXT:  %[[BYTE_OFFSET:.*]] = llvm.mul %[[COORDINATE]], %[[DIM_1_MEM_STRIDE_VAL]]  : i64
// CHECK-NEXT:  %[[SUBOJECT_OFFSET:.*]] = llvm.add %[[BYTE_OFFSET]], %[[OFFSET_INIT]]  : i64
// CHECK-NEXT:  %[[ARRAY_OBJECT_AS_VOID_PTR:.*]] = llvm.bitcast %[[ARRAY_OBJECT]] : !llvm.ptr<f32> to !llvm.ptr<i8>
// CHECK-NEXT:  %[[SUBOBJECT_ADDR:.*]] = llvm.getelementptr %[[ARRAY_OBJECT_AS_VOID_PTR]][%[[SUBOJECT_OFFSET]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
// CHECK-NEXT:  %[[RETURN_VAL:.*]] = llvm.bitcast %[[SUBOBJECT_ADDR]] : !llvm.ptr<i8> to !llvm.ptr<f32>
// CHECK-NEXT:  llvm.return

// -----

// `fir.array` inside a `fir.box` (2d)
func.func @coordinate_box_array_2d(%arg0: !fir.box<!fir.array<10 x 10 x f32>>, %arg1: index, %arg2: index) {
  %p = fir.coordinate_of %arg0, %arg1, %arg2 : (!fir.box<!fir.array<10 x 10 x f32>>, index, index) -> !fir.ref<f32>
  return
}
// CHECK-LABEL: llvm.func @coordinate_box_array_2d
// CHECK-SAME: %[[BOX:.*]]: !llvm.ptr<struct<(ptr<array<10 x array<10 x f32>>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<2 x array<3 x i64>>)>>
// CHECK-SAME: %[[COORDINATE_1:.*]]: i64, %[[COORDINATE_2:.*]]: i64)
// CHECK-NEXT:   %{{.*}} = llvm.mlir.constant(0 : i64) : i64
// There's only one box here. Its index is `0`. Generate it.
// CHECK-NEXT:  %[[BOX_IDX:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:  %[[BOX_1ST_ELEM_IDX:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:  %[[ARRAY_ADDR:.*]] = llvm.getelementptr %[[BOX]][%[[BOX_IDX]], 0] : (!llvm.ptr<struct<(ptr<array<10 x array<10 x f32>>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<2 x array<3 x i64>>)>>, i32) -> !llvm.ptr<ptr<array<10 x array<10 x f32>>>>
// CHECK-NEXT:  %[[ARRAY_OBJECT:.*]] = llvm.load %[[ARRAY_ADDR]] : !llvm.ptr<ptr<array<10 x array<10 x f32>>>>
// CHECK-NEXT:  %[[OFFSET_INIT:.*]] = llvm.mlir.constant(0 : i64) : i64
// Same as [[BOX_IDX]], just recreated.
// CHECK-NEXT:  %[[BOX_IDX_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// Index of the array that contains the CFI_dim_t objects
// CHECK-NEXT:  %[[CFI_DIM_IDX:.*]] = llvm.mlir.constant(7 : i32) : i32
// Index of the 1st CFI_dim_t object (corresonds the the 1st dimension)
// CHECK-NEXT:  %[[DIM_1_IDX:.*]] = llvm.mlir.constant(0 : i64) : i64
// Index of the memory stride within a CFI_dim_t object
// CHECK-NEXT:  %[[DIM_1_MEM_STRIDE:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:  %[[DIM_1_MEM_STRIDE_ADDR:.*]] = llvm.getelementptr %[[BOX]][%[[BOX_IDX_1]], 7, %[[DIM_1_IDX]], %[[DIM_1_MEM_STRIDE]]] : (!llvm.ptr<struct<(ptr<array<10 x array<10 x f32>>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<2 x array<3 x i64>>)>>, i32, i64, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:  %[[DIM_1_MEM_STRIDE_VAL:.*]] = llvm.load %[[DIM_1_MEM_STRIDE_ADDR]] : !llvm.ptr<i64>
// CHECK-NEXT:  %[[BYTE_OFFSET_1:.*]] = llvm.mul %[[COORDINATE_1]], %[[DIM_1_MEM_STRIDE_VAL]]  : i64
// CHECK-NEXT:  %[[SUBOBJECT_OFFSET_1:.*]] = llvm.add %[[BYTE_OFFSET]], %[[OFFSET_INIT]]  : i64
// Same as [[BOX_IDX]], just recreated.
// CHECK-NEXT:  %[[BOX_IDX_2:.*]] = llvm.mlir.constant(0 : i32) : i32
// Index of the array that contains the CFI_dim_t objects (same as CFI_DIM_IDX, just recreated)
// CHECK-NEXT:  %[[CFI_DIM_IDX_1:.*]] = llvm.mlir.constant(7 : i32) : i32
// Index of the 1st CFI_dim_t object (corresonds the the 2nd dimension)
// CHECK-NEXT:  %[[DIM_2_IDX:.*]] = llvm.mlir.constant(1 : i64) : i64
// Index of the memory stride within a CFI_dim_t object
// CHECK-NEXT:  %[[DIM_2_MEM_STRIDE:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:  %[[DIM_2_MEM_STRIDE_ADDR:.*]] = llvm.getelementptr %[[BOX]][%[[BOX_IDX_2]], 7, %[[DIM_2_IDX]], %[[DIM_2_MEM_STRIDE]]] : (!llvm.ptr<struct<(ptr<array<10 x array<10 x f32>>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<2 x array<3 x i64>>)>>, i32, i64, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:  %[[DIM_2_MEM_STRIDE_VAL:.*]] = llvm.load %[[DIM_2_MEM_STRIDE_ADDR]] : !llvm.ptr<i64>
// CHECK-NEXT:  %[[BYTE_OFFSET_2:.*]] = llvm.mul %[[COORDINATE_2]], %[[DIM_2_MEM_STRIDE_VAL]]  : i64
// CHECK-NEXT:  %[[SUBOBJECT_OFFSET_2:.*]] = llvm.add %[[BYTE_OFFSET_2]], %[[SUBOBJECT_OFFSET_1]]  : i64
// CHECK-NEXT:  %[[ARRAY_OBJECT_AS_VOID_PTR:.*]] = llvm.bitcast %[[ARRAY_OBJECT]] : !llvm.ptr<array<10 x array<10 x f32>>> to !llvm.ptr<i8>
// CHECK-NEXT:  %[[SUBOBJECT_ADDR:.*]] = llvm.getelementptr %[[ARRAY_OBJECT_AS_VOID_PTR]][%[[SUBOBJECT_OFFSET_2]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
// CHECK-NEXT:  %[[RETURN_VAL:.*]] = llvm.bitcast %[[SUBOBJECT_ADDR]] : !llvm.ptr<i8> to !llvm.ptr<f32>
// CHECK-NEXT:  llvm.return

// -----

// Test `fir.coordinate_of` conversion (items inside `!fir.box`)

// 4. BOX TYPE - `fir.derived` inside `fir.array`
func.func @coordinate_box_derived_inside_array(%arg0: !fir.box<!fir.array<10 x !fir.type<derived_3{field_1:f32, field_2:f32}>>>, %arg1 : index) {
   %idx0 = fir.field_index field_2, !fir.type<derived_3{field_1:f32, field_2:f32}>
   %q = fir.coordinate_of %arg0, %arg1, %idx0 : (!fir.box<!fir.array<10 x !fir.type<derived_3{field_1:f32, field_2:f32}>>>, index, !fir.field) -> !fir.ref<f32>
   return
}
// CHECK-LABEL:   llvm.func @coordinate_box_derived_inside_array(
// CHECK-SAME:    %[[BOX:.*]]: !llvm.ptr<struct<(ptr<array<10 x struct<"derived_3", (f32, f32)>>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>>,
// CHECK-SAME:    %[[COORDINATE_1:.*]]: i64) {
// CHECK:         %[[VAL_3:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[VAL_4:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[VAL_5:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[VAL_6:.*]] = llvm.getelementptr %[[BOX]]{{\[}}%[[VAL_4]], 0] : (!llvm.ptr<struct<(ptr<array<10 x struct<"derived_3", (f32, f32)>>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>>, i32) -> !llvm.ptr<ptr<array<10 x struct<"derived_3", (f32, f32)>>>>
// CHECK:         %[[ARRAY:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr<ptr<array<10 x struct<"derived_3", (f32, f32)>>>>
// CHECK:         %[[VAL_8:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[VAL_9:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[DIM_IDX:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[DIM_MEM_STRIDE:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:         %[[VAL_13:.*]] = llvm.getelementptr %[[BOX]][%[[VAL_9]], 7, %[[DIM_IDX]], %[[DIM_MEM_STRIDE]]] : (!llvm.ptr<struct<(ptr<array<10 x struct<"derived_3", (f32, f32)>>>, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>, ptr<i8>, array<1 x i64>)>>, i32, i64, i32) -> !llvm.ptr<i64>
// CHECK:         %[[VAL_14:.*]] = llvm.load %[[VAL_13]] : !llvm.ptr<i64>
// CHECK:         %[[VAL_15:.*]] = llvm.mul %[[COORDINATE_1]], %[[VAL_14]]  : i64
// CHECK:         %[[OFFSET:.*]] = llvm.add %[[VAL_15]], %[[VAL_8]]  : i64
// CHECK:         %[[VAL_17:.*]] = llvm.bitcast %[[ARRAY]] : !llvm.ptr<array<10 x struct<"derived_3", (f32, f32)>>> to !llvm.ptr<i8>
// CHECK:         %[[VAL_18:.*]] = llvm.getelementptr %[[VAL_17]][%[[OFFSET]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
// CHECK:         %[[DERIVED:.*]] = llvm.bitcast %[[VAL_18]] : !llvm.ptr<i8> to !llvm.ptr<struct<"derived_3", (f32, f32)>>
// CHECK:         %[[VAL_20:.*]] = llvm.getelementptr %[[DERIVED]][%[[VAL_3]], 1] : (!llvm.ptr<struct<"derived_3", (f32, f32)>>, i64) -> !llvm.ptr<f32>
// CHECK:         %[[VAL_21:.*]] = llvm.bitcast %[[VAL_20]] : !llvm.ptr<f32> to !llvm.ptr<i8>
// CHECK:         %[[VAL_22:.*]] = llvm.bitcast %[[VAL_21]] : !llvm.ptr<i8> to !llvm.ptr<f32>
// CHECK:         llvm.return

// -----

// Test `fir.coordinate_of` conversion (items inside `!fir.ref`)

// 5.1. `fir.array`
func.func @coordinate_array_unknown_size_1d(%arg0: !fir.ref<!fir.array<? x i32>>, %arg1 : index) {
   %q = fir.coordinate_of %arg0, %arg1 : (!fir.ref<!fir.array<? x i32>>, index) -> !fir.ref<i32>
   return
}
// CHECK-LABEL:   llvm.func @coordinate_array_unknown_size_1d(
// CHECK-SAME:    %[[VAL_0:.*]]: !llvm.ptr<i32>,
// CHECK-SAME:    %[[VAL_1:.*]]: i64) {
// CHECK:           %[[VAL_2:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
// CHECK:           llvm.return
// CHECK:         }

// -----

func.func @coordinate_array_known_size_1d(%arg0: !fir.ref<!fir.array<10 x i32>>, %arg1 : index) {
   %q = fir.coordinate_of %arg0, %arg1 : (!fir.ref<!fir.array<10 x i32>>, index) -> !fir.ref<i32>
   return
}
// CHECK-LABEL:   llvm.func @coordinate_array_known_size_1d(
// CHECK-SAME:    %[[VAL_0:.*]]: !llvm.ptr<array<10 x i32>>,
// CHECK-SAME:    %[[VAL_1:.*]]: i64) {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[VAL_3:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_2]], %[[VAL_1]]] : (!llvm.ptr<array<10 x i32>>, i64, i64) -> !llvm.ptr<i32>
// CHECK:           llvm.return
// CHECK:         }

// -----

func.func @coordinate_array_known_size_2d_get_i32(%arg0: !fir.ref<!fir.array<10 x 10 x i32>>, %arg1 : index, %arg2 : index) {
   %q = fir.coordinate_of %arg0, %arg1, %arg2 : (!fir.ref<!fir.array<10 x 10 x i32>>, index, index) -> !fir.ref<i32>
   return
}
// CHECK-LABEL:   llvm.func @coordinate_array_known_size_2d_get_i32(
// CHECK-SAME:    %[[VAL_0:.*]]: !llvm.ptr<array<10 x array<10 x i32>>>,
// CHECK-SAME:    %[[VAL_1:.*]]: i64,
// CHECK-SAME:    %[[VAL_2:.*]]: i64) {
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[VAL_4:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_3]], %[[VAL_2]], %[[VAL_1]]] : (!llvm.ptr<array<10 x array<10 x i32>>>, i64, i64, i64) -> !llvm.ptr<i32>
// CHECK:           llvm.return
// CHECK:         }

// -----

func.func @coordinate_array_known_size_2d_get_array(%arg0: !fir.ref<!fir.array<10 x 10 x i32>>, %arg1 : index) {
   %q = fir.coordinate_of %arg0, %arg1 : (!fir.ref<!fir.array<10 x 10 x i32>>, index) -> !fir.ref<!fir.array<10 x i32>>
   return
}
// CHECK-LABEL:   llvm.func @coordinate_array_known_size_2d_get_array(
// CHECK-SAME:    %[[VAL_0:.*]]: !llvm.ptr<array<10 x array<10 x i32>>>,
// CHECK-SAME:    %[[VAL_1:.*]]: i64) {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[VAL_3:.*]] = llvm.getelementptr %[[VAL_0]][%[[VAL_2]], %[[VAL_1]]] : (!llvm.ptr<array<10 x array<10 x i32>>>, i64, i64) -> !llvm.ptr<array<10 x i32>>
// CHECK:           llvm.return
// CHECK:         }

// -----

// 5.2. `fir.derived`
func.func @coordinate_ref_derived(%arg0: !fir.ref<!fir.type<dervied_4{field_1:i32, field_2:i32}>>) {
  %idx = fir.field_index field_2, !fir.type<dervied_4{field_1:i32, field_2:i32}>
  %q = fir.coordinate_of %arg0, %idx : (!fir.ref<!fir.type<dervied_4{field_1:i32, field_2:i32}>>, !fir.field) -> !fir.ref<i32>
  return
}
// CHECK-LABEL:   llvm.func @coordinate_ref_derived(
// CHECK-SAME:    %[[VAL_0:.*]]: !llvm.ptr<struct<"dervied_4", (i32, i32)>>) {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[VAL_3:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_2]], 1] : (!llvm.ptr<struct<"dervied_4", (i32, i32)>>, i64) -> !llvm.ptr<i32>
// CHECK:           llvm.return
// CHECK:         }

// -----

func.func @coordinate_ref_derived_nested(%arg0: !fir.ref<!fir.type<derived_5{field_1:!fir.type<nested_derived{inner1:i32, inner2:f32}>, field_2:i32}>>) {
  %idx0 = fir.field_index field_1, !fir.type<derived_5{field_1:!fir.type<nested_derived{inner1:i32, inner2:f32}>, field_2:i32}>
  %idx1 = fir.field_index inner2, !fir.type<nested_derived{inner1:i32, inner2:f32}>
  %q = fir.coordinate_of %arg0, %idx0, %idx1 : (!fir.ref<!fir.type<derived_5{field_1:!fir.type<nested_derived{inner1:i32, inner2:f32}>, field_2:i32}>>, !fir.field, !fir.field) -> !fir.ref<i32>
  return
}
// CHECK-LABEL:   llvm.func @coordinate_ref_derived_nested(
// CHECK-SAME:    %[[VAL_0:.*]]: !llvm.ptr<struct<"derived_5", (struct<"nested_derived", (i32, f32)>, i32)>>) {
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[VAL_4:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_3]], 0, 1] : (!llvm.ptr<struct<"derived_5", (struct<"nested_derived", (i32, f32)>, i32)>>, i64) -> !llvm.ptr<i32>
// CHECK:           llvm.return
// CHECK:         }

// -----

// 5.3 `fir.char`
func.func @test_coordinate_of_char(%arr : !fir.ref<!fir.char<10, 2>>) {
  %1 = arith.constant 10 : i32
  %2 = fir.coordinate_of %arr, %1 : (!fir.ref<!fir.char<10, 2>>, i32) -> !fir.ref<i80>
  return
}
// CHECK-LABEL:   llvm.func @test_coordinate_of_char(
// CHECK-SAME:    %[[VAL_0:.*]]: !llvm.ptr<array<2 x i80>>) {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(10 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]]] : (!llvm.ptr<array<2 x i80>>, i32) -> !llvm.ptr<i80>
// CHECK:           llvm.return
// CHECK:         }

// -----

// 5.4 `mlir.tuple`
func.func @test_coordinate_of_tuple(%tup : !fir.ref<tuple<!fir.ref<i32>>>) {
  %1 = arith.constant 0 : i32
  %2 = fir.coordinate_of %tup, %1 : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.ref<i32>
  return
}
// CHECK-LABEL:   llvm.func @test_coordinate_of_tuple(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !llvm.ptr<struct<(ptr<i32>)>>) {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[VAL_3:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_2]], 0] : (!llvm.ptr<struct<(ptr<i32>)>>, i64) -> !llvm.ptr<i32>
// CHECK:           llvm.return
// CHECK:         }

// -----

// Test `fir.coordinate_of` conversion - items inside `!fir.ptr`. This should
// be almost identical to `!fir.ref` (i.e. it's the same code path in the code
// gen). Instead of duplicating the tests, only one for sanity-checking is added.

// 6.1. `fir.array`
func.func @coordinate_array_unknown_size_1d(%arg0: !fir.ptr<!fir.array<? x i32>>, %arg1 : index) {
   %q = fir.coordinate_of %arg0, %arg1 : (!fir.ptr<!fir.array<? x i32>>, index) -> !fir.ref<i32>
   return
}
// CHECK-LABEL:   llvm.func @coordinate_array_unknown_size_1d(
// CHECK-SAME:    %[[VAL_0:.*]]: !llvm.ptr<i32>,
// CHECK-SAME:    %[[VAL_1:.*]]: i64) {
// CHECK:           %[[VAL_2:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
// CHECK:           llvm.return
// CHECK:         }
