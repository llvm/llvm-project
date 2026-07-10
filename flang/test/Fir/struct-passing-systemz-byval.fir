// Test SystemZ ABI rewrite of struct passed by value (BIND(C), VALUE derived types).
// RUN: fir-opt --target-rewrite="target=s390x-unknown-linux-gnu" %s | FileCheck %s

module attributes {
  fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:128-v128:64-a:8:16-n32:64", llvm.target_triple = "s390x-unknown-linux-gnu" } {

// CHECK-LABEL: func.func private @test_i32(!fir.ref<!fir.type<t1{i:i32}>>
// CHECK-SAME:    {llvm.align = 8 : i32, llvm.byval = !fir.type<t1{i:i32}>})
func.func private @test_i32(%arg0: !fir.type<t1{i:i32}>)

// CHECK-LABEL: func.func @test_call_i32(
// CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.type<t1{i:i32}>>) {
// CHECK: %[[IN:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.type<t1{i:i32}>>
// CHECK: %[[STCK:.*]] = llvm.intr.stacksave : !llvm.ptr
// CHECK: %[[TMP:.*]] = fir.alloca !fir.type<t1{i:i32}>
// CHECK: fir.store %[[IN]] to %[[TMP]] : !fir.ref<!fir.type<t1{i:i32}>>
// CHECK: fir.call @test_i32(%[[TMP]]) : (!fir.ref<!fir.type<t1{i:i32}>> {llvm.align = 8 : i32, llvm.byval = !fir.type<t1{i:i32}>}) -> ()
// CHECK: llvm.intr.stackrestore %[[STCK]] : !llvm.ptr
// CHECK: return
func.func @test_call_i32(%arg0: !fir.ref<!fir.type<t1{i:i32}>>) {
  %in = fir.load %arg0 : !fir.ref<!fir.type<t1{i:i32}>>
  fir.call @test_i32(%in) : (!fir.type<t1{i:i32}>) -> ()
  return
}

// CHECK-LABEL: func.func private @test_i64(!fir.ref<!fir.type<t2{i:i64}>>
// CHECK-SAME:    {llvm.align = 8 : i32, llvm.byval = !fir.type<t2{i:i64}>})
func.func private @test_i64(%arg0: !fir.type<t2{i:i64}>)

// CHECK-LABEL: func.func private @test_f32_f32(!fir.ref<!fir.type<t3{r:f32,i:f32}>>
// CHECK-SAME:    {llvm.align = 8 : i32, llvm.byval = !fir.type<t3{r:f32,i:f32}>})
func.func private @test_f32_f32(%arg0: !fir.type<t3{r:f32,i:f32}>)

// CHECK-LABEL: func.func private @test_f64_f64(!fir.ref<!fir.type<t4{r:f64,i:f64}>>
// CHECK-SAME:    {llvm.align = 8 : i32, llvm.byval = !fir.type<t4{r:f64,i:f64}>})
func.func private @test_f64_f64(%arg0: !fir.type<t4{r:f64,i:f64}>)

// CHECK-LABEL: func.func private @test_large(!fir.ref<!fir.type<t5{i:!fir.array<8xi32>}>>
// CHECK-SAME:    {llvm.align = 8 : i32, llvm.byval = !fir.type<t5{i:!fir.array<8xi32>}>})
func.func private @test_large(%arg0: !fir.type<t5{i:!fir.array<8xi32>}>)

// CHECK-LABEL: func.func private @test_mixed(!fir.ref<!fir.type<t6{i:i32,r:f64}>>
// CHECK-SAME:    {llvm.align = 8 : i32, llvm.byval = !fir.type<t6{i:i32,r:f64}>})
func.func private @test_mixed(%arg0: !fir.type<t6{i:i32,r:f64}>)

// Struct with f128 field — alignment is 16, not clamped down to 8.
// CHECK-LABEL: func.func private @test_f128(!fir.ref<!fir.type<t7{r:f128}>>
// CHECK-SAME:    {llvm.align = 16 : i32, llvm.byval = !fir.type<t7{r:f128}>})
func.func private @test_f128(%arg0: !fir.type<t7{r:f128}>)

} // end module
