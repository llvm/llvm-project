// Test SystemZ ABI rewrite of struct returned by value (BIND(C), VALUE derived types).
// RUN: fir-opt --target-rewrite="target=s390x-unknown-linux-gnu" %s | FileCheck %s

// CHECK-LABEL: func.func private @test_i32(!fir.ref<!fir.type<t1{i:i32}>>
// CHECK-SAME:    {llvm.align = 8 : i32, llvm.sret = !fir.type<t1{i:i32}>})
func.func private @test_i32() -> !fir.type<t1{i:i32}>

// CHECK-LABEL: func.func @test_call_i32(
// CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.type<t1{i:i32}>>) {
func.func @test_call_i32(%arg0 : !fir.ref<!fir.type<t1{i:i32}>>) {
  // CHECK: %[[STCK:.*]] = llvm.intr.stacksave : !llvm.ptr
  // CHECK: %[[ARG:.*]] = fir.alloca !fir.type<t1{i:i32}>
  // CHECK: fir.call @test_i32(%[[ARG]]) : (!fir.ref<!fir.type<t1{i:i32}>> {llvm.align = 8 : i32, llvm.sret = !fir.type<t1{i:i32}>}) -> ()
  // CHECK: %[[CVT:.*]] = fir.convert %[[ARG]] : (!fir.ref<!fir.type<t1{i:i32}>>) -> !fir.ref<!fir.type<t1{i:i32}>>
  // CHECK: %[[LD:.*]] = fir.load %[[CVT]] : !fir.ref<!fir.type<t1{i:i32}>>
  // CHECK: llvm.intr.stackrestore %[[STCK]] : !llvm.ptr
  %out = fir.call @test_i32() : () -> !fir.type<t1{i:i32}>
  // CHECK: fir.store %[[LD]] to %[[ARG0]] : !fir.ref<!fir.type<t1{i:i32}>>
  fir.store %out to %arg0 : !fir.ref<!fir.type<t1{i:i32}>>
  // CHECK: return
  return
}

// CHECK-LABEL: func.func private @test_i64(!fir.ref<!fir.type<t2{i:i64}>>
// CHECK-SAME:    {llvm.align = 8 : i32, llvm.sret = !fir.type<t2{i:i64}>})
func.func private @test_i64() -> !fir.type<t2{i:i64}>

// CHECK-LABEL: func.func @test_call_i64(
// CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.type<t2{i:i64}>>) {
func.func @test_call_i64(%arg0 : !fir.ref<!fir.type<t2{i:i64}>>) {
  // CHECK: %[[STCK:.*]] = llvm.intr.stacksave : !llvm.ptr
  // CHECK: %[[ARG:.*]] = fir.alloca !fir.type<t2{i:i64}>
  // CHECK: fir.call @test_i64(%[[ARG]]) : (!fir.ref<!fir.type<t2{i:i64}>> {llvm.align = 8 : i32, llvm.sret = !fir.type<t2{i:i64}>}) -> ()
  // CHECK: %[[CVT:.*]] = fir.convert %[[ARG]] : (!fir.ref<!fir.type<t2{i:i64}>>) -> !fir.ref<!fir.type<t2{i:i64}>>
  // CHECK: %[[LD:.*]] = fir.load %[[CVT]] : !fir.ref<!fir.type<t2{i:i64}>>
  // CHECK: llvm.intr.stackrestore %[[STCK]] : !llvm.ptr
  %out = fir.call @test_i64() : () -> !fir.type<t2{i:i64}>
  // CHECK: fir.store %[[LD]] to %[[ARG0]] : !fir.ref<!fir.type<t2{i:i64}>>
  fir.store %out to %arg0 : !fir.ref<!fir.type<t2{i:i64}>>
  // CHECK: return
  return
}

// CHECK-LABEL: func.func private @test_f32_f32(!fir.ref<!fir.type<t3{r:f32,i:f32}>>
// CHECK-SAME:    {llvm.align = 8 : i32, llvm.sret = !fir.type<t3{r:f32,i:f32}>})
func.func private @test_f32_f32() -> !fir.type<t3{r:f32,i:f32}>

// CHECK-LABEL: func.func private @test_f64_f64(!fir.ref<!fir.type<t4{r:f64,i:f64}>>
// CHECK-SAME:    {llvm.align = 8 : i32, llvm.sret = !fir.type<t4{r:f64,i:f64}>})
func.func private @test_f64_f64() -> !fir.type<t4{r:f64,i:f64}>

// CHECK-LABEL: func.func private @test_large(!fir.ref<!fir.type<t5{i:!fir.array<8xi32>}>>
// CHECK-SAME:    {llvm.align = 8 : i32, llvm.sret = !fir.type<t5{i:!fir.array<8xi32>}>})
func.func private @test_large() -> !fir.type<t5{i:!fir.array<8xi32>}>

// CHECK-LABEL: func.func private @test_f128(!fir.ref<!fir.type<t7{r:f128}>>
// CHECK-SAME:    {llvm.align = 8 : i32, llvm.sret = !fir.type<t7{r:f128}>})
func.func private @test_f128() -> !fir.type<t7{r:f128}>

