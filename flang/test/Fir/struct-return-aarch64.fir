// Test AArch64 ABI rewrite of struct returned by value (BIND(C), VALUE derived types).
// RUN: fir-opt --target-rewrite="target=aarch64-unknown-linux-gnu" %s | FileCheck %s

!composite = !fir.type<t1{i:f32,j:i32,k:f32}>
// CHECK-LABEL: func.func private @test_composite() -> !fir.array<2xi64>
func.func private @test_composite() -> !composite
// CHECK-LABEL: func.func @test_call_composite(
// CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.type<t1{i:f32,j:i32,k:f32}>>)
func.func @test_call_composite(%arg0 : !fir.ref<!composite>) {
  // CHECK: %[[OUT:.*]] = fir.call @test_composite() : () -> !fir.array<2xi64>
  // CHECK: %[[STACK:.*]] = llvm.intr.stacksave : !llvm.ptr
  // CHECK: %[[ARR:.*]] = fir.alloca !fir.array<2xi64>
  // CHECK: fir.store %[[OUT]] to %[[ARR]] : !fir.ref<!fir.array<2xi64>>
  // CHECK: %[[CVT:.*]] = fir.convert %[[ARR]] : (!fir.ref<!fir.array<2xi64>>) -> !fir.ref<!fir.type<t1{i:f32,j:i32,k:f32}>>
  // CHECK: %[[LD:.*]] = fir.load %[[CVT]] : !fir.ref<!fir.type<t1{i:f32,j:i32,k:f32}>>
  // CHECK: llvm.intr.stackrestore %[[STACK]] : !llvm.ptr
  %out = fir.call @test_composite() : () -> !composite
  // CHECK: fir.store %[[LD]] to %[[ARG0]] : !fir.ref<!fir.type<t1{i:f32,j:i32,k:f32}>>
  fir.store %out to %arg0 : !fir.ref<!composite>
  // CHECK: return
  return
}

!hfa_f16 = !fir.type<t2{x:f16, y:f16}>
// CHECK-LABEL: func.func private @test_hfa_f16() -> !fir.type<t2{x:f16,y:f16}>
func.func private @test_hfa_f16() -> !hfa_f16
// CHECK-LABEL: func.func @test_call_hfa_f16(
// CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.type<t2{x:f16,y:f16}>>) {
func.func @test_call_hfa_f16(%arg0 : !fir.ref<!hfa_f16>) {
  // CHECK: %[[OUT:.*]] = fir.call @test_hfa_f16() : () -> !fir.type<t2{x:f16,y:f16}>
  // CHECK: %[[STACK:.*]] = llvm.intr.stacksave : !llvm.ptr
  // CHECK: %[[ARR:.*]] = fir.alloca !fir.type<t2{x:f16,y:f16}>
  // CHECK: fir.store %[[OUT]] to %[[ARR]] : !fir.ref<!fir.type<t2{x:f16,y:f16}>>
  // CHECK: %[[CVT:.*]] = fir.convert %[[ARR]] : (!fir.ref<!fir.type<t2{x:f16,y:f16}>>) -> !fir.ref<!fir.type<t2{x:f16,y:f16}>>
  // CHECK: %[[LD:.*]] = fir.load %[[CVT]] : !fir.ref<!fir.type<t2{x:f16,y:f16}>>
  // CHECK: llvm.intr.stackrestore %[[STACK]] : !llvm.ptr
  %out = fir.call @test_hfa_f16() : () -> !hfa_f16
  // CHECK: fir.store %[[LD]] to %[[ARG0]] : !fir.ref<!fir.type<t2{x:f16,y:f16}>>
  fir.store %out to %arg0 : !fir.ref<!hfa_f16>
  return
}

!hfa_f32 = !fir.type<t3{w:f32, x:f32, y:f32, z:f32}>
// CHECK-LABEL: func.func private @test_hfa_f32() -> !fir.type<t3{w:f32,x:f32,y:f32,z:f32}>
func.func private @test_hfa_f32() -> !hfa_f32
// CHECK-LABEL: func.func @test_call_hfa_f32(
// CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.type<t3{w:f32,x:f32,y:f32,z:f32}>>) {
func.func @test_call_hfa_f32(%arg0 : !fir.ref<!hfa_f32>) {
  // CHECK: %[[OUT:.*]] = fir.call @test_hfa_f32() : () -> !fir.type<t3{w:f32,x:f32,y:f32,z:f32}>
  // CHECK: %[[STACK:.*]] = llvm.intr.stacksave : !llvm.ptr
  // CHECK: %[[ARR:.*]] = fir.alloca !fir.type<t3{w:f32,x:f32,y:f32,z:f32}>
  // CHECK: fir.store %[[OUT]] to %[[ARR]] : !fir.ref<!fir.type<t3{w:f32,x:f32,y:f32,z:f32}>>
  // CHECK: %[[CVT:.*]] = fir.convert %[[ARR]] : (!fir.ref<!fir.type<t3{w:f32,x:f32,y:f32,z:f32}>>) -> !fir.ref<!fir.type<t3{w:f32,x:f32,y:f32,z:f32}>>
  // CHECK: %[[LD:.*]] = fir.load %[[CVT]] : !fir.ref<!fir.type<t3{w:f32,x:f32,y:f32,z:f32}>>
  // CHECK: llvm.intr.stackrestore %[[STACK]] : !llvm.ptr
  %out = fir.call @test_hfa_f32() : () -> !hfa_f32
  // CHECK: fir.store %[[LD]] to %[[ARG0]] : !fir.ref<!fir.type<t3{w:f32,x:f32,y:f32,z:f32}>>
  fir.store %out to %arg0 : !fir.ref<!hfa_f32>
  return
}

!hfa_f64 = !fir.type<t4{x:f64, y:f64, z:f64}>
// CHECK-LABEL: func.func private @test_hfa_f64() -> !fir.type<t4{x:f64,y:f64,z:f64}>
func.func private @test_hfa_f64() -> !hfa_f64
// CHECK-LABEL: func.func @test_call_hfa_f64(
// CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.type<t4{x:f64,y:f64,z:f64}>>)
func.func @test_call_hfa_f64(%arg0 : !fir.ref<!hfa_f64>) {
  // CHECK: %[[OUT:.*]] = fir.call @test_hfa_f64() : () -> !fir.type<t4{x:f64,y:f64,z:f64}>
  // CHECK: %[[STACK:.*]] = llvm.intr.stacksave : !llvm.ptr
  // CHECK: %[[ARR:.*]] = fir.alloca !fir.type<t4{x:f64,y:f64,z:f64}>
  // CHECK: fir.store %[[OUT]] to %[[ARR]] : !fir.ref<!fir.type<t4{x:f64,y:f64,z:f64}>>
  // CHECK: %[[CVT:.*]] = fir.convert %[[ARR]] : (!fir.ref<!fir.type<t4{x:f64,y:f64,z:f64}>>) -> !fir.ref<!fir.type<t4{x:f64,y:f64,z:f64}>>
  // CHECK: %[[LD:.*]] = fir.load %[[CVT]] : !fir.ref<!fir.type<t4{x:f64,y:f64,z:f64}>>
  // CHECK: llvm.intr.stackrestore %[[STACK]] : !llvm.ptr
  %out = fir.call @test_hfa_f64() : () -> !hfa_f64
  // CHECK: fir.store %[[LD]] to %[[ARG0]] : !fir.ref<!fir.type<t4{x:f64,y:f64,z:f64}>>
  fir.store %out to %arg0 : !fir.ref<!hfa_f64>
  return
}

!hfa_f128 = !fir.type<t5{w:f128, x:f128, y:f128, z:f128}>
// CHECK-LABEL: func.func private @test_hfa_f128() -> !fir.type<t5{w:f128,x:f128,y:f128,z:f128}>
func.func private @test_hfa_f128() -> !hfa_f128
// CHECK-LABEL: func.func @test_call_hfa_f128(
// CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.type<t5{w:f128,x:f128,y:f128,z:f128}>>) {
func.func @test_call_hfa_f128(%arg0 : !fir.ref<!hfa_f128>) {
  // CHECK: %[[OUT:.*]] = fir.call @test_hfa_f128() : () -> !fir.type<t5{w:f128,x:f128,y:f128,z:f128}>
  // CHECK: %[[STACK:.*]] = llvm.intr.stacksave : !llvm.ptr
  // CHECK: %[[ARR:.*]] = fir.alloca !fir.type<t5{w:f128,x:f128,y:f128,z:f128}>
  // CHECK: fir.store %[[OUT]] to %[[ARR]] : !fir.ref<!fir.type<t5{w:f128,x:f128,y:f128,z:f128}>>
  // CHECK: %[[CVT:.*]] = fir.convert %[[ARR]] : (!fir.ref<!fir.type<t5{w:f128,x:f128,y:f128,z:f128}>>) -> !fir.ref<!fir.type<t5{w:f128,x:f128,y:f128,z:f128}>>
  // CHECK: %[[LD:.*]] = fir.load %[[CVT]] : !fir.ref<!fir.type<t5{w:f128,x:f128,y:f128,z:f128}>>
  // CHECK: llvm.intr.stackrestore %[[STACK]] : !llvm.ptr
  %out = fir.call @test_hfa_f128() : () -> !hfa_f128
  // CHECK: fir.store %[[LD]] to %[[ARG0]] : !fir.ref<!fir.type<t5{w:f128,x:f128,y:f128,z:f128}>>
  fir.store %out to %arg0 : !fir.ref<!hfa_f128>
  return
}

!hfa_bf16 = !fir.type<t6{w:bf16, x:bf16, y:bf16, z:bf16}>
// CHECK-LABEL: func.func private @test_hfa_bf16() -> !fir.type<t6{w:bf16,x:bf16,y:bf16,z:bf16}>
func.func private @test_hfa_bf16() -> !hfa_bf16
// CHECK-LABEL: func.func @test_call_hfa_bf16(
// CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.type<t6{w:bf16,x:bf16,y:bf16,z:bf16}>>) {
func.func @test_call_hfa_bf16(%arg0 : !fir.ref<!hfa_bf16>) {
  // CHECK: %[[OUT:.*]] = fir.call @test_hfa_bf16() : () -> !fir.type<t6{w:bf16,x:bf16,y:bf16,z:bf16}>
  // CHECK: %[[STACK:.*]] = llvm.intr.stacksave : !llvm.ptr
  // CHECK: %[[ARR:.*]] = fir.alloca !fir.type<t6{w:bf16,x:bf16,y:bf16,z:bf16}>
  // CHECK: fir.store %[[OUT]] to %[[ARR]] : !fir.ref<!fir.type<t6{w:bf16,x:bf16,y:bf16,z:bf16}>>
  // CHECK: %[[CVT:.*]] = fir.convert %[[ARR]] : (!fir.ref<!fir.type<t6{w:bf16,x:bf16,y:bf16,z:bf16}>>) -> !fir.ref<!fir.type<t6{w:bf16,x:bf16,y:bf16,z:bf16}>>
  // CHECK: %[[LD:.*]] = fir.load %[[CVT]] : !fir.ref<!fir.type<t6{w:bf16,x:bf16,y:bf16,z:bf16}>>
  // CHECK: llvm.intr.stackrestore %[[STACK]] : !llvm.ptr
  %out = fir.call @test_hfa_bf16() : () -> !hfa_bf16
  // CHECK: fir.store %[[LD]] to %[[ARG0]] : !fir.ref<!fir.type<t6{w:bf16,x:bf16,y:bf16,z:bf16}>>
  fir.store %out to %arg0 : !fir.ref<!hfa_bf16>
  return
}

!too_big = !fir.type<t7{x:i64, y:i64, z:i64}>
// CHECK-LABEL: func.func private @test_too_big(!fir.ref<!fir.type<t7{x:i64,y:i64,z:i64}>>
// CHECK-SAME:    {llvm.align = 8 : i32, llvm.sret = !fir.type<t7{x:i64,y:i64,z:i64}>})
func.func private @test_too_big() -> !too_big
// CHECK-LABEL: func.func @test_call_too_big(
// CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.type<t7{x:i64,y:i64,z:i64}>>) {
func.func @test_call_too_big(%arg0 : !fir.ref<!too_big>) {
  // CHECK: %[[STACK:.*]] = llvm.intr.stacksave : !llvm.ptr
  // CHECK: %[[ARG:.*]] = fir.alloca !fir.type<t7{x:i64,y:i64,z:i64}>
  // CHECK: fir.call @test_too_big(%[[ARG]]) : (!fir.ref<!fir.type<t7{x:i64,y:i64,z:i64}>>) -> ()
  // CHECK: %[[CVT:.*]] = fir.convert %[[ARG]] : (!fir.ref<!fir.type<t7{x:i64,y:i64,z:i64}>>) -> !fir.ref<!fir.type<t7{x:i64,y:i64,z:i64}>>
  // CHECK: %[[LD:.*]] = fir.load %[[CVT]] : !fir.ref<!fir.type<t7{x:i64,y:i64,z:i64}>>
  // CHECK: llvm.intr.stackrestore %[[STACK]] : !llvm.ptr
  %out = fir.call @test_too_big() : () -> !too_big
  // CHECK: fir.store %[[LD]] to %[[ARG0]] : !fir.ref<!fir.type<t7{x:i64,y:i64,z:i64}>>
  fir.store %out to %arg0 : !fir.ref<!too_big>
  return
}


!too_big_hfa = !fir.type<t8{i:!fir.array<5xf32>}>
// CHECK-LABEL: func.func private @test_too_big_hfa(!fir.ref<!fir.type<t8{i:!fir.array<5xf32>}>>
// CHECK-SAME:    {llvm.align = 8 : i32, llvm.sret = !fir.type<t8{i:!fir.array<5xf32>}>})
func.func private @test_too_big_hfa() -> !too_big_hfa
// CHECK-LABEL: func.func @test_call_too_big_hfa(
// CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.type<t8{i:!fir.array<5xf32>}>>) {
func.func @test_call_too_big_hfa(%arg0 : !fir.ref<!too_big_hfa>) {
  // CHECK: %[[STACK:.*]] = llvm.intr.stacksave : !llvm.ptr
  // CHECK: %[[ARG:.*]] = fir.alloca !fir.type<t8{i:!fir.array<5xf32>}>
  // CHECK: fir.call @test_too_big_hfa(%[[ARG]]) : (!fir.ref<!fir.type<t8{i:!fir.array<5xf32>}>>) -> ()
  // CHECK: %[[CVT:.*]] = fir.convert %[[ARG]] : (!fir.ref<!fir.type<t8{i:!fir.array<5xf32>}>>) -> !fir.ref<!fir.type<t8{i:!fir.array<5xf32>}>>
  // CHECK: %[[LD:.*]] = fir.load %[[CVT]] : !fir.ref<!fir.type<t8{i:!fir.array<5xf32>}>>
  // CHECK: llvm.intr.stackrestore %[[STACK]] : !llvm.ptr
  %out = fir.call @test_too_big_hfa() : () -> !too_big_hfa
  // CHECK: fir.store %[[LD]] to %[[ARG0]] : !fir.ref<!fir.type<t8{i:!fir.array<5xf32>}>>
  fir.store %out to %arg0 : !fir.ref<!too_big_hfa>
  return
}

!nested_hfa_first = !fir.type<t9{s:!hfa_f16,c:f16}>
// CHECK-LABEL: func.func private @test_nested_hfa_first() -> !fir.type<t9{s:!fir.type<t2{x:f16,y:f16}>,c:f16}>
func.func private @test_nested_hfa_first() -> !nested_hfa_first
// CHECK-LABEL: func.func @test_call_nested_hfa_first(%arg0: !fir.ref<!fir.type<t9{s:!fir.type<t2{x:f16,y:f16}>,c:f16}>>) {
func.func @test_call_nested_hfa_first(%arg0 : !fir.ref<!nested_hfa_first>) {
  %out = fir.call @test_nested_hfa_first() : () -> !nested_hfa_first
  // CHECK: %[[OUT:.*]] = fir.call @test_nested_hfa_first() : () -> !fir.type<t9{s:!fir.type<t2{x:f16,y:f16}>,c:f16}>
  // CHECK: %[[STACK:.*]] = llvm.intr.stacksave : !llvm.ptr
  // CHECK: %[[ARR:.*]] = fir.alloca !fir.type<t9{s:!fir.type<t2{x:f16,y:f16}>,c:f16}>
  // CHECK: fir.store %[[OUT]] to %[[ARR]] : !fir.ref<!fir.type<t9{s:!fir.type<t2{x:f16,y:f16}>,c:f16}>>
  // CHECK: %[[CVT:.*]] = fir.convert %[[ARR]] : (!fir.ref<!fir.type<t9{s:!fir.type<t2{x:f16,y:f16}>,c:f16}>>
  // CHECK: %[[LD:.*]] = fir.load %[[CVT]] : !fir.ref<!fir.type<t9{s:!fir.type<t2{x:f16,y:f16}>,c:f16}>>
  // CHECK: llvm.intr.stackrestore %[[STACK]] : !llvm.ptr
  fir.store %out to %arg0 : !fir.ref<!nested_hfa_first>
  // CHECK fir.store %[[LD]] to %[[ARG0]] : !fir.ref<!fir.type<t9{s:!fir.type<t2{x:f16,y:f16}>,c:f16}>>
  return
}


!nested_hfa_middle = !fir.type<t10{a:f16,s:!hfa_f16,c:f16}>
// CHECK-LABEL: func.func private @test_nested_hfa_middle() -> !fir.type<t10{a:f16,s:!fir.type<t2{x:f16,y:f16}>,c:f16}>
func.func private @test_nested_hfa_middle() -> !nested_hfa_middle
// CHECK-LABEL: func.func @test_call_nested_hfa_middle(%arg0: !fir.ref<!fir.type<t10{a:f16,s:!fir.type<t2{x:f16,y:f16}>,c:f16}>>) {
func.func @test_call_nested_hfa_middle(%arg0 : !fir.ref<!nested_hfa_middle>) {
  %out = fir.call @test_nested_hfa_middle() : () -> !nested_hfa_middle
  // CHECK: %[[OUT:.*]] = fir.call @test_nested_hfa_middle() : () -> !fir.type<t10{a:f16,s:!fir.type<t2{x:f16,y:f16}>,c:f16}>
  // CHECK: %[[STACK:.*]] = llvm.intr.stacksave : !llvm.ptr
  // CHECK: %[[ARR:.*]] = fir.alloca !fir.type<t10{a:f16,s:!fir.type<t2{x:f16,y:f16}>,c:f16}>
  // CHECK: fir.store %[[OUT]] to %[[ARR]] : !fir.ref<!fir.type<t10{a:f16,s:!fir.type<t2{x:f16,y:f16}>,c:f16}>>
  // CHECK: %[[CVT:.*]] = fir.convert %[[ARR]] : (!fir.ref<!fir.type<t10{a:f16,s:!fir.type<t2{x:f16,y:f16}>,c:f16}>>
  // CHECK: %[[LD:.*]] = fir.load %[[CVT]] : !fir.ref<!fir.type<t10{a:f16,s:!fir.type<t2{x:f16,y:f16}>,c:f16}>>
  // CHECK: llvm.intr.stackrestore %[[STACK]] : !llvm.ptr
  fir.store %out to %arg0 : !fir.ref<!nested_hfa_middle>
  // CHECK fir.store %[[LD]] to %[[ARG0]] : !fir.ref<!fir.type<t10{a:f16,s:!fir.type<t2{x:f16,y:f16}>,c:f16}>>
  return
}

!nested_hfa_end = !fir.type<t11{a:f16,s:!hfa_f16}>
// CHECK-LABEL: func.func private @test_nested_hfa_end() -> !fir.type<t11{a:f16,s:!fir.type<t2{x:f16,y:f16}>}>
func.func private @test_nested_hfa_end() -> !nested_hfa_end
// CHECK-LABEL: func.func @test_call_nested_hfa_end(%arg0: !fir.ref<!fir.type<t11{a:f16,s:!fir.type<t2{x:f16,y:f16}>}>>) {
func.func @test_call_nested_hfa_end(%arg0 : !fir.ref<!nested_hfa_end>) {
  %out = fir.call @test_nested_hfa_end() : () -> !nested_hfa_end
  // CHECK: %[[OUT:.*]] = fir.call @test_nested_hfa_end() : () -> !fir.type<t11{a:f16,s:!fir.type<t2{x:f16,y:f16}>}>
  // CHECK: %[[STACK:.*]] = llvm.intr.stacksave : !llvm.ptr
  // CHECK: %[[ARR:.*]] = fir.alloca !fir.type<t11{a:f16,s:!fir.type<t2{x:f16,y:f16}>}>
  // CHECK: fir.store %[[OUT]] to %[[ARR]] : !fir.ref<!fir.type<t11{a:f16,s:!fir.type<t2{x:f16,y:f16}>}>>
  // CHECK: %[[CVT:.*]] = fir.convert %[[ARR]] : (!fir.ref<!fir.type<t11{a:f16,s:!fir.type<t2{x:f16,y:f16}>}>>
  // CHECK: %[[LD:.*]] = fir.load %[[CVT]] : !fir.ref<!fir.type<t11{a:f16,s:!fir.type<t2{x:f16,y:f16}>}>>
  // CHECK: llvm.intr.stackrestore %[[STACK]] : !llvm.ptr
  fir.store %out to %arg0 : !fir.ref<!nested_hfa_end>
  // CHECK fir.store %[[LD]] to %[[ARG0]] : !fir.ref<!fir.type<t11{a:f16,s:!fir.type<t2{x:f16,y:f16}>}>>
  return
}

!nested_hfa_array = !fir.type<t12{a:!fir.array<2xf32>,b:f32}>
// CHECK-LABEL: func.func private @test_nested_hfa_array() -> !fir.type<t12{a:!fir.array<2xf32>,b:f32}>
func.func private @test_nested_hfa_array() -> !nested_hfa_array
// CHECK-LABEL: func.func @test_call_nested_hfa_array(%arg0: !fir.ref<!fir.type<t12{a:!fir.array<2xf32>,b:f32}>
func.func @test_call_nested_hfa_array(%arg0 : !fir.ref<!nested_hfa_array>) {
  %out = fir.call @test_nested_hfa_array() : () -> !nested_hfa_array
  // CHECK: %[[OUT:.*]] = fir.call @test_nested_hfa_array() : () -> !fir.type<t12{a:!fir.array<2xf32>,b:f32}>
  // CHECK: %[[STACK:.*]] = llvm.intr.stacksave : !llvm.ptr
  // CHECK: %[[ARR:.*]] = fir.alloca !fir.type<t12{a:!fir.array<2xf32>,b:f32}>
  // CHECK: fir.store %[[OUT]] to %[[ARR]] : !fir.ref<!fir.type<t12{a:!fir.array<2xf32>,b:f32}>
  // CHECK: %[[CVT:.*]] = fir.convert %[[ARR]] : (!fir.ref<!fir.type<t12{a:!fir.array<2xf32>,b:f32}>
  // CHECK: %[[LD:.*]] = fir.load %[[CVT]] : !fir.ref<!fir.type<t12{a:!fir.array<2xf32>,b:f32}>
  // CHECK: llvm.intr.stackrestore %[[STACK]] : !llvm.ptr
  fir.store %out to %arg0 : !fir.ref<!nested_hfa_array>
  // CHECK fir.store %[[LD]] to %[[ARG0]] : !fir.ref<!fir.type<t12{a:!fir.array<2xf32>,b:f32}>
  return
}
