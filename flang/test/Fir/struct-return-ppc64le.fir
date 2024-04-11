// Test ppc64le ABI rewrite of struct returned by value (BIND(C), VALUE derived types).
//
// RUN: fir-opt --target-rewrite="target=ppc64le-ibm-linux" %s | FileCheck %s
// REQUIRES: powerpc-registered-target

// character type
!t1 = !fir.type<t1{c:!fir.char<1>}>
func.func private @test_t1() -> !t1
//CHECK-LABEL: func.func private @test_t1(!fir.ref<!fir.type<t1{c:!fir.char<1>}>> {llvm.align = 8 : i32, llvm.sret = !fir.type<t1{c:!fir.char<1>}>})
func.func @test_call_t1(%arg0 : !fir.ref<!t1>) {
//CHECK-LABEL: func.func @test_call_t1(
//CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.type<t1{c:!fir.char<1>}>>)
  %out = fir.call @test_t1() : () -> !t1
  fir.store %out to %arg0 : !fir.ref<!t1>
  return
  //CHECK: %[[STCK:.*]] = llvm.intr.stacksave : !llvm.ptr
  //CHECK: %[[ARG:.*]] = fir.alloca !fir.type<t1{c:!fir.char<1>}>
  //CHECK: fir.call @test_t1(%[[ARG]]) : (!fir.ref<!fir.type<t1{c:!fir.char<1>}>>) -> ()

  //CHECK: %[[CVT:.*]] = fir.convert %[[ARG]] : (!fir.ref<!fir.type<t1{c:!fir.char<1>}>>) -> !fir.ref<!fir.type<t1{c:!fir.char<1>}>>
  //CHECK: %[[LD:.*]] = fir.load %[[CVT]] : !fir.ref<!fir.type<t1{c:!fir.char<1>}>>
  //CHECK: llvm.intr.stackrestore %[[STCK]] : !llvm.ptr
  //CHECK: fir.store %[[LD]] to %[[ARG0]] : !fir.ref<!fir.type<t1{c:!fir.char<1>}>>
  //CHECK: return
}

// integer type
!t2 = !fir.type<t2{i:i32}>
func.func private @test_t2() -> !t2
//CHECK-LABEL: func.func private @test_t2(!fir.ref<!fir.type<t2{i:i32}>> {llvm.align = 8 : i32, llvm.sret = !fir.type<t2{i:i32}>})
func.func @test_call_t2(%arg0 : !fir.ref<!t2>) {
//CHECK-LABEL: func.func @test_call_t2(
//CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.type<t2{i:i32}>>)
  %out = fir.call @test_t2() : () -> !t2
  fir.store %out to %arg0 : !fir.ref<!t2>
  return
  //CHECK: %[[STCK:.*]] = llvm.intr.stacksave : !llvm.ptr
  //CHECK: %[[ARG:.*]] = fir.alloca !fir.type<t2{i:i32}>
  //CHECK: fir.call @test_t2(%[[ARG]]) : (!fir.ref<!fir.type<t2{i:i32}>>) -> ()

  //CHECK: %[[CVT:.*]] = fir.convert %[[ARG]] : (!fir.ref<!fir.type<t2{i:i32}>>) -> !fir.ref<!fir.type<t2{i:i32}>>
  //CHECK: %[[LD:.*]] = fir.load %[[CVT]] : !fir.ref<!fir.type<t2{i:i32}>>
  //CHECK: llvm.intr.stackrestore %[[STCK]] : !llvm.ptr
  //CHECK: fir.store %[[LD]] to %[[ARG0]] : !fir.ref<!fir.type<t2{i:i32}>>
  //CHECK: return
}

// real type (scalar)
!t3 = !fir.type<t3{r1:f32,r2:f32,r3:f32}>
func.func private @test_t3() -> !t3
//CHECK-LABEL: func.func private @test_t3() -> !fir.array<3xf32>
func.func @test_call_t3(%arg0 : !fir.ref<!t3>) {
//CHECK-LABEL: func.func @test_call_t3(
//CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.type<t3{r1:f32,r2:f32,r3:f32}>>)
  %out = fir.call @test_t3() : () -> !t3
  fir.store %out to %arg0 : !fir.ref<!t3>
  return
  //CHECK: %[[RET:.*]] = fir.call @test_t3() : () -> !fir.array<3xf32>
  //CHECK: %[[STCK:.*]] = llvm.intr.stacksave : !llvm.ptr
  //CHECK: %[[ARG:.*]] = fir.alloca !fir.array<3xf32>
  //CHECK: fir.store %[[RET]] to %[[ARG]] : !fir.ref<!fir.array<3xf32>>
  //CHECK: %[[CVT:.*]] = fir.convert %[[ARG]] : (!fir.ref<!fir.array<3xf32>>) -> !fir.ref<!fir.type<t3{r1:f32,r2:f32,r3:f32}>>
  //CHECK: %[[LD:.*]] = fir.load %[[CVT]] : !fir.ref<!fir.type<t3{r1:f32,r2:f32,r3:f32}>>
  //CHECK: llvm.intr.stackrestore %[[STCK]] : !llvm.ptr
  //CHECK: fir.store %[[LD]] to %[[ARG0]] : !fir.ref<!fir.type<t3{r1:f32,r2:f32,r3:f32}>>
  //CHECK: return
}

// real type (> 8 floats)
!t4 = !fir.type<t4{r:!fir.array<9xf32>}>
func.func private @test_t4() -> !t4
//CHECK-LABEL: func.func private @test_t4(!fir.ref<!fir.type<t4{r:!fir.array<9xf32>}>> {llvm.align = 8 : i32, llvm.sret = !fir.type<t4{r:!fir.array<9xf32>}>})
func.func @test_call_t4(%arg0 : !fir.ref<!t4>) {
//CHECK-LABEL: func.func @test_call_t4(
//CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.type<t4{r:!fir.array<9xf32>}>>)
  %out = fir.call @test_t4() : () -> !t4
  fir.store %out to %arg0 : !fir.ref<!t4>
  return
  //CHECK: %[[STCK:.*]] = llvm.intr.stacksave : !llvm.ptr
  //CHECK: %[[ARG:.*]] = fir.alloca !fir.type<t4{r:!fir.array<9xf32>}>
  //CHECK: fir.call @test_t4(%[[ARG]]) : (!fir.ref<!fir.type<t4{r:!fir.array<9xf32>}>>) -> ()
  //CHECK: %[[CVT:.*]] = fir.convert %[[ARG]] : (!fir.ref<!fir.type<t4{r:!fir.array<9xf32>}>>) -> !fir.ref<!fir.type<t4{r:!fir.array<9xf32>}>>
  //CHECK: %[[LD:.*]] = fir.load %[[CVT]] : !fir.ref<!fir.type<t4{r:!fir.array<9xf32>}>>
  //CHECK: llvm.intr.stackrestore %[[STCK]] : !llvm.ptr
  //CHECK: fir.store %[[LD]] to %[[ARG0]] : !fir.ref<!fir.type<t4{r:!fir.array<9xf32>}>>
  //CHECK: return
}

// mixed types
!t5 = !fir.type<t5{c:!fir.char<1>,r:f32,i:i64}>
func.func private @test_t5() -> !t5
//CHECK-LABEL: func.func private @test_t5(!fir.ref<!fir.type<t5{c:!fir.char<1>,r:f32,i:i64}>> {llvm.align = 8 : i32, llvm.sret = !fir.type<t5{c:!fir.char<1>,r:f32,i:i64}>})
func.func @test_call_t5(%arg0 : !fir.ref<!t5>) {
//CHECK-LABEL: func.func @test_call_t5(
//CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.type<t5{c:!fir.char<1>,r:f32,i:i64}>>)
  %out = fir.call @test_t5() : () -> !t5
  fir.store %out to %arg0 : !fir.ref<!t5>
  return
  //CHECK: %[[STCK:.*]] = llvm.intr.stacksave : !llvm.ptr
  //CHECK: %[[ARG:.*]] = fir.alloca !fir.type<t5{c:!fir.char<1>,r:f32,i:i64}>
  //CHECK: fir.call @test_t5(%[[ARG]]) : (!fir.ref<!fir.type<t5{c:!fir.char<1>,r:f32,i:i64}>>) -> ()
  //CHECK: %[[CVT:.*]] = fir.convert %[[ARG]] : (!fir.ref<!fir.type<t5{c:!fir.char<1>,r:f32,i:i64}>>) -> !fir.ref<!fir.type<t5{c:!fir.char<1>,r:f32,i:i64}>>
  //CHECK: %[[LD:.*]] = fir.load %[[CVT]] : !fir.ref<!fir.type<t5{c:!fir.char<1>,r:f32,i:i64}>>
  //CHECK: llvm.intr.stackrestore %[[STCK]] : !llvm.ptr
  //CHECK: fir.store %[[LD]] to %[[ARG0]] : !fir.ref<!fir.type<t5{c:!fir.char<1>,r:f32,i:i64}>>
  //CHECK: return
}
