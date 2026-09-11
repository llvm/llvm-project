// Test SystemZ ABI rewrite of struct passed by reference (BIND(C), VALUE derived types).
// RUN: fir-opt --target-rewrite="target=s390x-unknown-linux-gnu" %s | FileCheck %s

// CHECK-LABEL: func.func private @test_i32(%arg0: i32)
func.func private @test_i32(%arg0: !fir.type<t1{i:i32}>) { return }

// CHECK-LABEL: func.func @test_call_i32(
// CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.type<t1{i:i32}>>) {
func.func @test_call_i32(%arg0: !fir.ref<!fir.type<t1{i:i32}>>) {
  // CHECK: %[[IN:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.type<t1{i:i32}>>
  %in = fir.load %arg0 : !fir.ref<!fir.type<t1{i:i32}>>
  // CHECK: %[[STCK:.*]] = llvm.intr.stacksave : !llvm.ptr
  // CHECK: %[[TMP:.*]] = fir.alloca i32
  // CHECK: %[[CVT:.*]] = fir.convert %[[TMP]] : (!fir.ref<i32>) -> !fir.ref<!fir.type<t1{i:i32}>>
  // CHECK: fir.store %[[IN]] to %[[CVT]] : !fir.ref<!fir.type<t1{i:i32}>>
  // CHECK: %[[VAL:.*]] = fir.load %[[TMP]] : !fir.ref<i32>
  // CHECK: fir.call @test_i32(%[[VAL]]) : (i32) -> ()
  // CHECK: llvm.intr.stackrestore %[[STCK]] : !llvm.ptr
  fir.call @test_i32(%in) : (!fir.type<t1{i:i32}>) -> ()
  // CHECK: return
  return
}

// CHECK-LABEL: func.func private @test_i64(%arg0: i64)
func.func private @test_i64(%arg0: !fir.type<t2{i:i64}>) { return }

// CHECK-LABEL: func.func private @test_f32_f32(%arg0: i64)
func.func private @test_f32_f32(%arg0: !fir.type<t3{r:f32,i:f32}>) { return }

// CHECK-LABEL: func.func private @test_f64_f64(%arg0: !fir.ref<!fir.type<t4{r:f64,i:f64}>>
// CHECK-SAME:    {llvm.align = 8 : i32})
func.func private @test_f64_f64(%arg0: !fir.type<t4{r:f64,i:f64}>) { return }

// CHECK-LABEL: func.func private @test_large(%arg0: !fir.ref<!fir.type<t5{i:!fir.array<8xi32>}>>
// CHECK-SAME:    {llvm.align = 8 : i32})
func.func private @test_large(%arg0: !fir.type<t5{i:!fir.array<8xi32>}>) { return }

// CHECK-LABEL: func.func private @test_mixed(%arg0: !fir.ref<!fir.type<t6{i:i32,r:f64}>>
// CHECK-SAME:    {llvm.align = 8 : i32})
func.func private @test_mixed(%arg0: !fir.type<t6{i:i32,r:f64}>) { return }

// CHECK-LABEL: func.func private @test_f128(%arg0: !fir.ref<!fir.type<t7{r:f128}>>
// CHECK-SAME:    {llvm.align = 8 : i32})
func.func private @test_f128(%arg0: !fir.type<t7{r:f128}>) { return }

// CHECK-LABEL: func.func private @test_f32(%arg0: f32)
func.func private @test_f32(%arg0: !fir.type<t8{r:f32}>) { return }

// CHECK-LABEL: func.func @test_call_f32(
// CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.type<t8{r:f32}>>) {
func.func @test_call_f32(%arg0: !fir.ref<!fir.type<t8{r:f32}>>) {
  // CHECK: %[[IN:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.type<t8{r:f32}>>
  %in = fir.load %arg0 : !fir.ref<!fir.type<t8{r:f32}>>
  // CHECK: %[[STCK:.*]] = llvm.intr.stacksave : !llvm.ptr
  // CHECK: %[[TMP:.*]] = fir.alloca f32
  // CHECK: %[[CVT:.*]] = fir.convert %[[TMP]] : (!fir.ref<f32>) -> !fir.ref<!fir.type<t8{r:f32}>>
  // CHECK: fir.store %[[IN]] to %[[CVT]] : !fir.ref<!fir.type<t8{r:f32}>>
  // CHECK: %[[VAL:.*]] = fir.load %[[TMP]] : !fir.ref<f32>
  // CHECK: fir.call @test_f32(%[[VAL]]) : (f32) -> ()
  // CHECK: llvm.intr.stackrestore %[[STCK]] : !llvm.ptr
  fir.call @test_f32(%in) : (!fir.type<t8{r:f32}>) -> ()
  // CHECK: return
  return
}

// CHECK-LABEL: func.func private @test_f64(%arg0: f64)
func.func private @test_f64(%arg0: !fir.type<t9{r:f64}>) { return }

// CHECK-LABEL: func.func @test_call_f64(
// CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.type<t9{r:f64}>>) {
func.func @test_call_f64(%arg0: !fir.ref<!fir.type<t9{r:f64}>>) {
  // CHECK: %[[IN:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.type<t9{r:f64}>>
  %in = fir.load %arg0 : !fir.ref<!fir.type<t9{r:f64}>>
  // CHECK: %[[STCK:.*]] = llvm.intr.stacksave : !llvm.ptr
  // CHECK: %[[TMP:.*]] = fir.alloca f64
  // CHECK: %[[CVT:.*]] = fir.convert %[[TMP]] : (!fir.ref<f64>) -> !fir.ref<!fir.type<t9{r:f64}>>
  // CHECK: fir.store %[[IN]] to %[[CVT]] : !fir.ref<!fir.type<t9{r:f64}>>
  // CHECK: %[[VAL:.*]] = fir.load %[[TMP]] : !fir.ref<f64>
  // CHECK: fir.call @test_f64(%[[VAL]]) : (f64) -> ()
  // CHECK: llvm.intr.stackrestore %[[STCK]] : !llvm.ptr
  fir.call @test_f64(%in) : (!fir.type<t9{r:f64}>) -> ()
  // CHECK: return
  return
}

// CHECK-LABEL: func.func private @test_nested_f64(%arg0: f64)
func.func private @test_nested_f64(%arg0: !fir.type<t10{s:!fir.type<t9{r:f64}>}>) { return }

// Large struct call: no byval, passes address.
// CHECK-LABEL: func.func @test_call_large(
// CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.type<t5{i:!fir.array<8xi32>}>>) {
func.func @test_call_large(%arg0: !fir.ref<!fir.type<t5{i:!fir.array<8xi32>}>>) {
  // CHECK: %[[IN:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.type<t5{i:!fir.array<8xi32>}>>
  %in = fir.load %arg0 : !fir.ref<!fir.type<t5{i:!fir.array<8xi32>}>>
  // CHECK: %[[STCK:.*]] = llvm.intr.stacksave : !llvm.ptr
  // CHECK: %[[TMP:.*]] = fir.alloca !fir.type<t5{i:!fir.array<8xi32>}>
  // CHECK: fir.store %[[IN]] to %[[TMP]] : !fir.ref<!fir.type<t5{i:!fir.array<8xi32>}>>
  // CHECK: fir.call @test_large(%[[TMP]]) : (!fir.ref<!fir.type<t5{i:!fir.array<8xi32>}>> {llvm.align = 8 : i32}) -> ()
  // CHECK: llvm.intr.stackrestore %[[STCK]] : !llvm.ptr
  fir.call @test_large(%in) : (!fir.type<t5{i:!fir.array<8xi32>}>) -> ()
  // CHECK: return
  return
}

