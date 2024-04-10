// Test X86-64 passing ABI of struct in registers for the cases where the
// struct has more than one field.
// REQUIRES: x86-registered-target
// RUN: fir-opt -target-rewrite="target=x86_64-unknown-linux-gnu" %s -o - | FileCheck %s


module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {

func.func @test_call_i8_a16(%0 : !fir.ref<!fir.type<ti8_a16{a:!fir.array<16xi8>}>>) {
  %1 = fir.load %0 : !fir.ref<!fir.type<ti8_a16{a:!fir.array<16xi8>}>>
  fir.call @test_func_i8_a16(%1)  : (!fir.type<ti8_a16{a:!fir.array<16xi8>}>) -> ()
  return
}
// CHECK-LABEL:   func.func @test_call_i8_a16(
// CHECK-SAME:                                %[[VAL_0:.*]]: !fir.ref<!fir.type<ti8_a16{a:!fir.array<16xi8>}>>) {
// CHECK:           %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.type<ti8_a16{a:!fir.array<16xi8>}>>
// CHECK:           %[[VAL_2:.*]] = fir.call @llvm.stacksave.p0() : () -> !fir.ref<i8>
// CHECK:           %[[VAL_3:.*]] = fir.alloca tuple<i64, i64>
// CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<tuple<i64, i64>>) -> !fir.ref<!fir.type<ti8_a16{a:!fir.array<16xi8>}>>
// CHECK:           fir.store %[[VAL_1]] to %[[VAL_4]] : !fir.ref<!fir.type<ti8_a16{a:!fir.array<16xi8>}>>
// CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_3]] : !fir.ref<tuple<i64, i64>>
// CHECK:           %[[VAL_6:.*]] = fir.extract_value %[[VAL_5]], [0 : i32] : (tuple<i64, i64>) -> i64
// CHECK:           %[[VAL_7:.*]] = fir.extract_value %[[VAL_5]], [1 : i32] : (tuple<i64, i64>) -> i64
// CHECK:           fir.call @test_func_i8_a16(%[[VAL_6]], %[[VAL_7]]) : (i64, i64) -> ()
// CHECK:           fir.call @llvm.stackrestore.p0(%[[VAL_2]]) : (!fir.ref<i8>) -> ()
// CHECK:           return

func.func private @test_func_i8_a16(%0 : !fir.type<ti8_a16{a:!fir.array<16xi8>}>) -> () {
  return
}
// CHECK-LABEL:   func.func private @test_func_i8_a16(
// CHECK-SAME:                                        %[[VAL_0:.*]]: i64,
// CHECK-SAME:                                        %[[VAL_1:.*]]: i64) {
// CHECK:           %[[VAL_2:.*]] = fir.undefined tuple<i64, i64>
// CHECK:           %[[VAL_3:.*]] = fir.insert_value %[[VAL_2]], %[[VAL_0]], [0 : i32] : (tuple<i64, i64>, i64) -> tuple<i64, i64>
// CHECK:           %[[VAL_4:.*]] = fir.insert_value %[[VAL_3]], %[[VAL_1]], [1 : i32] : (tuple<i64, i64>, i64) -> tuple<i64, i64>
// CHECK:           %[[VAL_5:.*]] = fir.alloca tuple<i64, i64>
// CHECK:           fir.store %[[VAL_4]] to %[[VAL_5]] : !fir.ref<tuple<i64, i64>>
// CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<tuple<i64, i64>>) -> !fir.ref<!fir.type<ti8_a16{a:!fir.array<16xi8>}>>
// CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_6]] : !fir.ref<!fir.type<ti8_a16{a:!fir.array<16xi8>}>>
// CHECK:           return


// For the cases below, the argument marshalling logic is the same as above,
// so only the chosen signature is tested at the end.

func.func @test_call_i32_f32(%0 : !fir.ref<!fir.type<ti32_f32{i:i32,x:f32}>>) {
  %1 = fir.load %0 : !fir.ref<!fir.type<ti32_f32{i:i32,x:f32}>>
  fir.call @test_func_i32_f32(%1)  : (!fir.type<ti32_f32{i:i32,x:f32}>) -> ()
  return
}
func.func private @test_func_i32_f32(%0 : !fir.type<ti32_f32{i:i32,x:f32}>) -> () {
  return
}

func.func @test_call_i32_i16(%0 : !fir.ref<!fir.type<ti32_i16{i:i32,x:i16}>>) {
  %1 = fir.load %0 : !fir.ref<!fir.type<ti32_i16{i:i32,x:i16}>>
  fir.call @test_func_i32_i16(%1)  : (!fir.type<ti32_i16{i:i32,x:i16}>) -> ()
  return
}
func.func private @test_func_i32_i16(%0 : !fir.type<ti32_i16{i:i32,x:i16}>) -> () {
  return
}

func.func @test_call_f16_i16(%0 : !fir.ref<!fir.type<tf16_i16{i:f16,x:i16}>>) {
  %1 = fir.load %0 : !fir.ref<!fir.type<tf16_i16{i:f16,x:i16}>>
  fir.call @test_func_f16_i16(%1)  : (!fir.type<tf16_i16{i:f16,x:i16}>) -> ()
  return
}
func.func private @test_func_f16_i16(%0 : !fir.type<tf16_i16{i:f16,x:i16}>) -> () {
  return
}

func.func @test_call_f16_f16(%0 : !fir.ref<!fir.type<tf16_f16{i:f16,x:f16}>>) {
  %1 = fir.load %0 : !fir.ref<!fir.type<tf16_f16{i:f16,x:f16}>>
  fir.call @test_func_f16_f16(%1)  : (!fir.type<tf16_f16{i:f16,x:f16}>) -> ()
  return
}
func.func private @test_func_f16_f16(%0 : !fir.type<tf16_f16{i:f16,x:f16}>) -> () {
  return
}

func.func @test_call_i32_f64(%0 : !fir.ref<!fir.type<ti32_f64{i:i32,x:f64}>>) {
  %1 = fir.load %0 : !fir.ref<!fir.type<ti32_f64{i:i32,x:f64}>>
  fir.call @test_func_i32_f64(%1)  : (!fir.type<ti32_f64{i:i32,x:f64}>) -> ()
  return
}
func.func private @test_func_i32_f64(%0 : !fir.type<ti32_f64{i:i32,x:f64}>) -> () {
  return
}

func.func @test_call_f64_f32(%0 : !fir.ref<!fir.type<tf64_f32{i:f64,x:f32}>>) {
  %1 = fir.load %0 : !fir.ref<!fir.type<tf64_f32{i:f64,x:f32}>>
  fir.call @test_func_f64_f32(%1)  : (!fir.type<tf64_f32{i:f64,x:f32}>) -> ()
  return
}
func.func private @test_func_f64_f32(%0 : !fir.type<tf64_f32{i:f64,x:f32}>) -> () {
  return
}

func.func @test_call_f32_i32_f32_f32(%0 : !fir.ref<!fir.type<tf32_i32_f32_f32{i:i32,x:f32,y:f32,z:f32}>>) {
  %1 = fir.load %0 : !fir.ref<!fir.type<tf32_i32_f32_f32{i:i32,x:f32,y:f32,z:f32}>>
  fir.call @test_func_f32_i32_f32_f32(%1)  : (!fir.type<tf32_i32_f32_f32{i:i32,x:f32,y:f32,z:f32}>) -> ()
  return
}
func.func private @test_func_f32_i32_f32_f32(%0 : !fir.type<tf32_i32_f32_f32{i:i32,x:f32,y:f32,z:f32}>) -> () {
  return
}

func.func @test_call_f64_i32(%before : i16, %0 : !fir.ref<!fir.type<tf64_i32{i:f64,x:i32}>>, %after : f128) {
  %1 = fir.load %0 : !fir.ref<!fir.type<tf64_i32{i:f64,x:i32}>>
  fir.call @test_func_f64_i32(%before, %1, %after)  : (i16, !fir.type<tf64_i32{i:f64,x:i32}>, f128) -> ()
  return
}
func.func private @test_func_f64_i32(%before : i16, %0 : !fir.type<tf64_i32{i:f64,x:i32}>, %after : f128) -> () {
  return
}
}

// CHECK-LABEL:   func.func @test_call_i32_f32(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.type<ti32_f32{i:i32,x:f32}>>) {
// CHECK-LABEL:   func.func private @test_func_i32_f32(
// CHECK-SAME:    %[[VAL_0:.*]]: i64) {
// CHECK-LABEL:   func.func @test_call_i32_i16(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.type<ti32_i16{i:i32,x:i16}>>) {
// CHECK-LABEL:   func.func private @test_func_i32_i16(
// CHECK-SAME:    %[[VAL_0:.*]]: i64) {
// CHECK-LABEL:   func.func @test_call_f16_i16(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.type<tf16_i16{i:f16,x:i16}>>) {
// CHECK-LABEL:   func.func private @test_func_f16_i16(
// CHECK-SAME:    %[[VAL_0:.*]]: i32) {
// CHECK-LABEL:   func.func @test_call_f16_f16(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.type<tf16_f16{i:f16,x:f16}>>) {
// CHECK-LABEL:   func.func private @test_func_f16_f16(
// CHECK-SAME:    %[[VAL_0:.*]]: f32) {
// CHECK-LABEL:   func.func @test_call_i32_f64(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.type<ti32_f64{i:i32,x:f64}>>) {
// CHECK-LABEL:   func.func private @test_func_i32_f64(
// CHECK-SAME:    %[[VAL_0:.*]]: i64,
// CHECK-SAME:    %[[VAL_1:.*]]: f64) {
// CHECK-LABEL:   func.func @test_call_f64_f32(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.type<tf64_f32{i:f64,x:f32}>>) {
// CHECK-LABEL:   func.func private @test_func_f64_f32(
// CHECK-SAME:    %[[VAL_0:.*]]: f64,
// CHECK-SAME:    %[[VAL_1:.*]]: f32) {
// CHECK-LABEL:   func.func @test_call_f32_i32_f32_f32(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.type<tf32_i32_f32_f32{i:i32,x:f32,y:f32,z:f32}>>) {
// CHECK-LABEL:   func.func private @test_func_f32_i32_f32_f32(
// CHECK-SAME:    %[[VAL_0:.*]]: i64,
// CHECK-SAME:    %[[VAL_1:.*]]: f64) {
// CHECK-LABEL:   func.func @test_call_f64_i32(
// CHECK-SAME:    %[[VAL_0:.*]]: i16,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<!fir.type<tf64_i32{i:f64,x:i32}>>,
// CHECK-SAME:    %[[VAL_2:.*]]: f128) {
// CHECK-LABEL:   func.func private @test_func_f64_i32(
// CHECK-SAME:    %[[VAL_0:.*]]: i16,
// CHECK-SAME:    %[[VAL_1:.*]]: f64,
// CHECK-SAME:    %[[VAL_2:.*]]: i32,
// CHECK-SAME:    %[[VAL_3:.*]]: f128) {
