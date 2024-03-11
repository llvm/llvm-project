// Test X86-64 passing ABI of struct in registers for the simple case
// where the struct has a single intrinsic component that is not a complex.
// REQUIRES: x86-registered-target
// RUN: fir-opt -target-rewrite="target=x86_64-unknown-linux-gnu" %s -o - | FileCheck %s

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {

func.func @test_call_i16(%0 : !fir.ref<!fir.type<ti16{i:i16}>>) {
  %7 = fir.load %0 : !fir.ref<!fir.type<ti16{i:i16}>>
  fir.call @test_func_i16(%7)  : (!fir.type<ti16{i:i16}>) -> ()
  return
}
// CHECK-LABEL:   func.func @test_call_i16(
// CHECK-SAME:                             %[[VAL_0:.*]]: !fir.ref<!fir.type<ti16{i:i16}>>) {
// CHECK:           %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.type<ti16{i:i16}>>
// CHECK:           %[[VAL_2:.*]] = fir.call @llvm.stacksave.p0() : () -> !fir.ref<i8>
// CHECK:           %[[VAL_3:.*]] = fir.alloca i16
// CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<i16>) -> !fir.ref<!fir.type<ti16{i:i16}>>
// CHECK:           fir.store %[[VAL_1]] to %[[VAL_4]] : !fir.ref<!fir.type<ti16{i:i16}>>
// CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_3]] : !fir.ref<i16>
// CHECK:           fir.call @test_func_i16(%[[VAL_5]]) : (i16) -> ()
// CHECK:           fir.call @llvm.stackrestore.p0(%[[VAL_2]]) : (!fir.ref<i8>) -> ()

func.func private @test_func_i16(%0 : !fir.type<ti16{i:i16}>) -> () {
  return
}
// CHECK-LABEL:   func.func private @test_func_i16(
// CHECK-SAME:                                     %[[VAL_0:.*]]: i16) {
// CHECK:           %[[VAL_1:.*]] = fir.alloca i16
// CHECK:           fir.store %[[VAL_0]] to %[[VAL_1]] : !fir.ref<i16>
// CHECK:           %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<i16>) -> !fir.ref<!fir.type<ti16{i:i16}>>
// CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.type<ti16{i:i16}>>

func.func @test_call_i32(%0 : !fir.ref<!fir.type<ti32{i:i32}>>) {
  %7 = fir.load %0 : !fir.ref<!fir.type<ti32{i:i32}>>
  fir.call @test_func_i32(%7)  : (!fir.type<ti32{i:i32}>) -> ()
  return
}
func.func private @test_func_i32(%0 : !fir.type<ti32{i:i32}>) -> () {
  return
}

func.func @test_call_i64(%0 : !fir.ref<!fir.type<ti64{i:i64}>>) {
  %7 = fir.load %0 : !fir.ref<!fir.type<ti64{i:i64}>>
  fir.call @test_func_i64(%7)  : (!fir.type<ti64{i:i64}>) -> ()
  return
}
func.func private @test_func_i64(%0 : !fir.type<ti64{i:i64}>) -> () {
  return
}

func.func @test_call_i128(%0 : !fir.ref<!fir.type<ti128{i:i128}>>) {
  %7 = fir.load %0 : !fir.ref<!fir.type<ti128{i:i128}>>
  fir.call @test_func_i128(%7)  : (!fir.type<ti128{i:i128}>) -> ()
  return
}
func.func private @test_func_i128(%0 : !fir.type<ti128{i:i128}>) -> () {
  return
}
func.func @test_call_f16(%0 : !fir.ref<!fir.type<tf16{i:f16}>>) {
  %7 = fir.load %0 : !fir.ref<!fir.type<tf16{i:f16}>>
  fir.call @test_func_f16(%7)  : (!fir.type<tf16{i:f16}>) -> ()
  return
}
func.func private @test_func_f16(%0 : !fir.type<tf16{i:f16}>) -> () {
  return
}

func.func @test_call_f32(%0 : !fir.ref<!fir.type<tf32{i:f32}>>) {
  %7 = fir.load %0 : !fir.ref<!fir.type<tf32{i:f32}>>
  fir.call @test_func_f32(%7)  : (!fir.type<tf32{i:f32}>) -> ()
  return
}
func.func private @test_func_f32(%0 : !fir.type<tf32{i:f32}>) -> () {
  return
}

func.func @test_call_f64(%0 : !fir.ref<!fir.type<tf64{i:f64}>>) {
  %7 = fir.load %0 : !fir.ref<!fir.type<tf64{i:f64}>>
  fir.call @test_func_f64(%7)  : (!fir.type<tf64{i:f64}>) -> ()
  return
}
func.func private @test_func_f64(%0 : !fir.type<tf64{i:f64}>) -> () {
  return
}

func.func @test_call_f128(%0 : !fir.ref<!fir.type<tf128{i:f128}>>) {
  %7 = fir.load %0 : !fir.ref<!fir.type<tf128{i:f128}>>
  fir.call @test_func_f128(%7)  : (!fir.type<tf128{i:f128}>) -> ()
  return
}
func.func private @test_func_f128(%0 : !fir.type<tf128{i:f128}>) -> () {
  return
}

func.func @test_call_char1(%0 : !fir.ref<!fir.type<tchar1{i:!fir.char<1>}>>) {
  %7 = fir.load %0 : !fir.ref<!fir.type<tchar1{i:!fir.char<1>}>>
  fir.call @test_func_char1(%7)  : (!fir.type<tchar1{i:!fir.char<1>}>) -> ()
  return
}
func.func private @test_func_char1(%0 : !fir.type<tchar1{i:!fir.char<1>}>) -> () {
  return
}

func.func @test_call_log1(%0 : !fir.ref<!fir.type<tlog1{i:!fir.logical<1>}>>) {
  %7 = fir.load %0 : !fir.ref<!fir.type<tlog1{i:!fir.logical<1>}>>
  fir.call @test_func_log1(%7)  : (!fir.type<tlog1{i:!fir.logical<1>}>) -> ()
  return
}
func.func private @test_func_log1(%0 : !fir.type<tlog1{i:!fir.logical<1>}>) -> () {
  return
}

func.func @test_call_log2(%0 : !fir.ref<!fir.type<tlog2{i:!fir.logical<2>}>>) {
  %7 = fir.load %0 : !fir.ref<!fir.type<tlog2{i:!fir.logical<2>}>>
  fir.call @test_func_log2(%7)  : (!fir.type<tlog2{i:!fir.logical<2>}>) -> ()
  return
}
func.func private @test_func_log2(%0 : !fir.type<tlog2{i:!fir.logical<2>}>) -> () {
  return
}

func.func @test_call_log4(%0 : !fir.ref<!fir.type<tlog4{i:!fir.logical<4>}>>) {
  %7 = fir.load %0 : !fir.ref<!fir.type<tlog4{i:!fir.logical<4>}>>
  fir.call @test_func_log4(%7)  : (!fir.type<tlog4{i:!fir.logical<4>}>) -> ()
  return
}
func.func private @test_func_log4(%0 : !fir.type<tlog4{i:!fir.logical<4>}>) -> () {
  return
}

func.func @test_call_log8(%0 : !fir.ref<!fir.type<tlog8{i:!fir.logical<8>}>>) {
  %7 = fir.load %0 : !fir.ref<!fir.type<tlog8{i:!fir.logical<8>}>>
  fir.call @test_func_log8(%7)  : (!fir.type<tlog8{i:!fir.logical<8>}>) -> ()
  return
}
func.func private @test_func_log8(%0 : !fir.type<tlog8{i:!fir.logical<8>}>) -> () {
  return
}

func.func @test_call_log16(%0 : !fir.ref<!fir.type<tlog16{i:!fir.logical<16>}>>) {
  %7 = fir.load %0 : !fir.ref<!fir.type<tlog16{i:!fir.logical<16>}>>
  fir.call @test_func_log16(%7)  : (!fir.type<tlog16{i:!fir.logical<16>}>) -> ()
  return
}
func.func private @test_func_log16(%0 : !fir.type<tlog16{i:!fir.logical<16>}>) -> () {
  return
}
}

// CHECK-LABEL:   func.func private @test_func_i32(
// CHECK-SAME:                                     %[[VAL_0:.*]]: i32) {
// CHECK-LABEL:   func.func private @test_func_i64(
// CHECK-SAME:                                     %[[VAL_0:.*]]: i64) {
// CHECK-LABEL:   func.func private @test_func_i128(
// CHECK-SAME:                                      %[[VAL_0:.*]]: i128) {
// CHECK-LABEL:   func.func private @test_func_f16(
// CHECK-SAME:                                     %[[VAL_0:.*]]: f16) {
// CHECK-LABEL:   func.func private @test_func_f32(
// CHECK-SAME:                                     %[[VAL_0:.*]]: f32) {
// CHECK-LABEL:   func.func private @test_func_f64(
// CHECK-SAME:                                     %[[VAL_0:.*]]: f64) {
// CHECK-LABEL:   func.func private @test_func_f128(
// CHECK-SAME:                                      %[[VAL_0:.*]]: f128) {
// CHECK-LABEL:   func.func private @test_func_char1(
// CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.char<1>) {
// CHECK-LABEL:   func.func private @test_func_log1(
// CHECK-SAME:                                      %[[VAL_0:.*]]: !fir.logical<1>) {
// CHECK-LABEL:   func.func private @test_func_log2(
// CHECK-SAME:                                      %[[VAL_0:.*]]: !fir.logical<2>) {
// CHECK-LABEL:   func.func private @test_func_log4(
// CHECK-SAME:                                      %[[VAL_0:.*]]: !fir.logical<4>) {
// CHECK-LABEL:   func.func private @test_func_log8(
// CHECK-SAME:                                      %[[VAL_0:.*]]: !fir.logical<8>) {
// CHECK-LABEL:   func.func private @test_func_log16(
// CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.logical<16>) {
