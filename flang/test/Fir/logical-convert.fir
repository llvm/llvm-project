// RUN: fir-opt --split-input-file --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s | FileCheck %s
// RUN: fir-opt --split-input-file --fir-to-llvm-ir="target=aarch64-unknown-linux-gnu" %s | FileCheck %s
// RUN: fir-opt --split-input-file --fir-to-llvm-ir="target=i386-unknown-linux-gnu" %s | FileCheck %s
// RUN: fir-opt --split-input-file --fir-to-llvm-ir="target=powerpc64le-unknown-linux-gn" %s | FileCheck %s

// -----
// CHECK-LABEL: @test_l1_i1
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i8
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i8
// CHECK: llvm.return [[t1]] : i1
func.func @test_l1_i1(%arg0: !fir.logical<1>) -> i1 {
  %0 = fir.convert %arg0 : (!fir.logical<1>) -> i1
  return %0 : i1
}
// -----
// CHECK-LABEL: @test_l1_i8
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i8
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i8
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i8
// CHECK: llvm.return [[t2]] : i8
func.func @test_l1_i8(%arg0: !fir.logical<1>) -> i8 {
  %0 = fir.convert %arg0 : (!fir.logical<1>) -> i8
  return %0 : i8
}
// -----
// CHECK-LABEL: @test_l1_i16
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i8
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i8
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i16
// CHECK: llvm.return [[t2]] : i16
func.func @test_l1_i16(%arg0: !fir.logical<1>) -> i16 {
  %0 = fir.convert %arg0 : (!fir.logical<1>) -> i16
  return %0 : i16
}
// -----
// CHECK-LABEL: @test_l1_i32
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i8
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i8
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i32
// CHECK: llvm.return [[t2]] : i32
func.func @test_l1_i32(%arg0: !fir.logical<1>) -> i32 {
  %0 = fir.convert %arg0 : (!fir.logical<1>) -> i32
  return %0 : i32
}
// -----
// CHECK-LABEL: @test_l1_i64
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i8
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i8
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i64
// CHECK: llvm.return [[t2]] : i64
func.func @test_l1_i64(%arg0: !fir.logical<1>) -> i64 {
  %0 = fir.convert %arg0 : (!fir.logical<1>) -> i64
  return %0 : i64
}
// -----
// CHECK-LABEL: @test_l2_i1
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i16
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i16
// CHECK: llvm.return [[t1]] : i1
func.func @test_l2_i1(%arg0: !fir.logical<2>) -> i1 {
  %0 = fir.convert %arg0 : (!fir.logical<2>) -> i1
  return %0 : i1
}
// -----
// CHECK-LABEL: @test_l2_i8
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i16
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i16
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i8
// CHECK: llvm.return [[t2]] : i8
func.func @test_l2_i8(%arg0: !fir.logical<2>) -> i8 {
  %0 = fir.convert %arg0 : (!fir.logical<2>) -> i8
  return %0 : i8
}
// -----
// CHECK-LABEL: @test_l2_i16
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i16
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i16
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i16
// CHECK: llvm.return [[t2]] : i16
func.func @test_l2_i16(%arg0: !fir.logical<2>) -> i16 {
  %0 = fir.convert %arg0 : (!fir.logical<2>) -> i16
  return %0 : i16
}
// -----
// CHECK-LABEL: @test_l2_i32
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i16
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i16
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i32
// CHECK: llvm.return [[t2]] : i32
func.func @test_l2_i32(%arg0: !fir.logical<2>) -> i32 {
  %0 = fir.convert %arg0 : (!fir.logical<2>) -> i32
  return %0 : i32
}
// -----
// CHECK-LABEL: @test_l2_i64
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i16
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i16
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i64
// CHECK: llvm.return [[t2]] : i64
func.func @test_l2_i64(%arg0: !fir.logical<2>) -> i64 {
  %0 = fir.convert %arg0 : (!fir.logical<2>) -> i64
  return %0 : i64
}
// -----
// CHECK-LABEL: @test_l4_i1
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i32
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i32
// CHECK: llvm.return [[t1]] : i1
func.func @test_l4_i1(%arg0: !fir.logical<4>) -> i1 {
  %0 = fir.convert %arg0 : (!fir.logical<4>) -> i1
  return %0 : i1
}
// -----
// CHECK-LABEL: @test_l4_i8
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i32
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i32
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i8
// CHECK: llvm.return [[t2]] : i8
func.func @test_l4_i8(%arg0: !fir.logical<4>) -> i8 {
  %0 = fir.convert %arg0 : (!fir.logical<4>) -> i8
  return %0 : i8
}
// -----
// CHECK-LABEL: @test_l4_i16
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i32
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i32
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i16
// CHECK: llvm.return [[t2]] : i16
func.func @test_l4_i16(%arg0: !fir.logical<4>) -> i16 {
  %0 = fir.convert %arg0 : (!fir.logical<4>) -> i16
  return %0 : i16
}
// -----
// CHECK-LABEL: @test_l4_i32
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i32
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i32
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i32
// CHECK: llvm.return [[t2]] : i32
func.func @test_l4_i32(%arg0: !fir.logical<4>) -> i32 {
  %0 = fir.convert %arg0 : (!fir.logical<4>) -> i32
  return %0 : i32
}
// -----
// CHECK-LABEL: @test_l4_i64
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i32
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i32
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i64
// CHECK: llvm.return [[t2]] : i64
func.func @test_l4_i64(%arg0: !fir.logical<4>) -> i64 {
  %0 = fir.convert %arg0 : (!fir.logical<4>) -> i64
  return %0 : i64
}
// -----
// CHECK-LABEL: @test_l8_i1
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i64
// CHECK: llvm.return [[t1]] : i1
func.func @test_l8_i1(%arg0: !fir.logical<8>) -> i1 {
  %0 = fir.convert %arg0 : (!fir.logical<8>) -> i1
  return %0 : i1
}
// -----
// CHECK-LABEL: @test_l8_i8
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i64
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i8
// CHECK: llvm.return [[t2]] : i8
func.func @test_l8_i8(%arg0: !fir.logical<8>) -> i8 {
  %0 = fir.convert %arg0 : (!fir.logical<8>) -> i8
  return %0 : i8
}
// -----
// CHECK-LABEL: @test_l8_i16
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i64
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i16
// CHECK: llvm.return [[t2]] : i16
func.func @test_l8_i16(%arg0: !fir.logical<8>) -> i16 {
  %0 = fir.convert %arg0 : (!fir.logical<8>) -> i16
  return %0 : i16
}
// -----
// CHECK-LABEL: @test_l8_i32
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i64
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i32
// CHECK: llvm.return [[t2]] : i32
func.func @test_l8_i32(%arg0: !fir.logical<8>) -> i32 {
  %0 = fir.convert %arg0 : (!fir.logical<8>) -> i32
  return %0 : i32
}
// -----
// CHECK-LABEL: @test_l8_i64
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i64
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i64
// CHECK: llvm.return [[t2]] : i64
func.func @test_l8_i64(%arg0: !fir.logical<8>) -> i64 {
  %0 = fir.convert %arg0 : (!fir.logical<8>) -> i64
  return %0 : i64
}
// -----
// CHECK-LABEL: @test_i1_l1
// CHECK: [[t0:%[0-9]*]] = llvm.zext %arg0 : i1 to i8
// CHECK: llvm.return [[t0]] : i8
func.func @test_i1_l1(%arg0: i1) -> !fir.logical<1> {
  %0 = fir.convert %arg0 : (i1) -> !fir.logical<1>
  return %0 : !fir.logical<1>
}
// -----
// CHECK-LABEL: @test_i1_l2
// CHECK: [[t0:%[0-9]*]] = llvm.zext %arg0 : i1 to i16
// CHECK: llvm.return [[t0]] : i16
func.func @test_i1_l2(%arg0: i1) -> !fir.logical<2> {
  %0 = fir.convert %arg0 : (i1) -> !fir.logical<2>
  return %0 : !fir.logical<2>
}
// -----
// CHECK-LABEL: @test_i1_l4
// CHECK: [[t0:%[0-9]*]] = llvm.zext %arg0 : i1 to i32
// CHECK: llvm.return [[t0]] : i32
func.func @test_i1_l4(%arg0: i1) -> !fir.logical<4> {
  %0 = fir.convert %arg0 : (i1) -> !fir.logical<4>
  return %0 : !fir.logical<4>
}
// -----
// CHECK-LABEL: @test_i1_l8
// CHECK: [[t0:%[0-9]*]] = llvm.zext %arg0 : i1 to i64
// CHECK: llvm.return [[t0]] : i64
func.func @test_i1_l8(%arg0: i1) -> !fir.logical<8> {
  %0 = fir.convert %arg0 : (i1) -> !fir.logical<8>
  return %0 : !fir.logical<8>
}
// -----
// CHECK-LABEL: @test_i8_l1
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i8
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i8
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i8
// CHECK: llvm.return [[t2]] : i8
func.func @test_i8_l1(%arg0: i8) -> !fir.logical<1> {
  %0 = fir.convert %arg0 : (i8) -> !fir.logical<1>
  return %0 : !fir.logical<1>
}
// -----
// CHECK-LABEL: @test_i8_l2
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i8
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i8
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i16
// CHECK: llvm.return [[t2]] : i16
func.func @test_i8_l2(%arg0: i8) -> !fir.logical<2> {
  %0 = fir.convert %arg0 : (i8) -> !fir.logical<2>
  return %0 : !fir.logical<2>
}
// -----
// CHECK-LABEL: @test_i8_l4
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i8
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i8
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i32
// CHECK: llvm.return [[t2]] : i32
func.func @test_i8_l4(%arg0: i8) -> !fir.logical<4> {
  %0 = fir.convert %arg0 : (i8) -> !fir.logical<4>
  return %0 : !fir.logical<4>
}
// -----
// CHECK-LABEL: @test_i8_l8
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i8
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i8
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i64
// CHECK: llvm.return [[t2]] : i64
func.func @test_i8_l8(%arg0: i8) -> !fir.logical<8> {
  %0 = fir.convert %arg0 : (i8) -> !fir.logical<8>
  return %0 : !fir.logical<8>
}
// -----
// CHECK-LABEL: @test_i16_l1
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i16
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i16
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i8
// CHECK: llvm.return [[t2]] : i8
func.func @test_i16_l1(%arg0: i16) -> !fir.logical<1> {
  %0 = fir.convert %arg0 : (i16) -> !fir.logical<1>
  return %0 : !fir.logical<1>
}
// -----
// CHECK-LABEL: @test_i16_l2
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i16
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i16
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i16
// CHECK: llvm.return [[t2]] : i16
func.func @test_i16_l2(%arg0: i16) -> !fir.logical<2> {
  %0 = fir.convert %arg0 : (i16) -> !fir.logical<2>
  return %0 : !fir.logical<2>
}
// -----
// CHECK-LABEL: @test_i16_l4
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i16
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i16
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i32
// CHECK: llvm.return [[t2]] : i32
func.func @test_i16_l4(%arg0: i16) -> !fir.logical<4> {
  %0 = fir.convert %arg0 : (i16) -> !fir.logical<4>
  return %0 : !fir.logical<4>
}
// -----
// CHECK-LABEL: @test_i16_l8
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i16
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i16
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i64
// CHECK: llvm.return [[t2]] : i64
func.func @test_i16_l8(%arg0: i16) -> !fir.logical<8> {
  %0 = fir.convert %arg0 : (i16) -> !fir.logical<8>
  return %0 : !fir.logical<8>
}
// -----
// CHECK-LABEL: @test_i32_l1
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i32
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i32
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i8
// CHECK: llvm.return [[t2]] : i8
func.func @test_i32_l1(%arg0: i32) -> !fir.logical<1> {
  %0 = fir.convert %arg0 : (i32) -> !fir.logical<1>
  return %0 : !fir.logical<1>
}
// -----
// CHECK-LABEL: @test_i32_l2
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i32
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i32
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i16
// CHECK: llvm.return [[t2]] : i16
func.func @test_i32_l2(%arg0: i32) -> !fir.logical<2> {
  %0 = fir.convert %arg0 : (i32) -> !fir.logical<2>
  return %0 : !fir.logical<2>
}
// -----
// CHECK-LABEL: @test_i32_l4
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i32
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i32
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i32
// CHECK: llvm.return [[t2]] : i32
func.func @test_i32_l4(%arg0: i32) -> !fir.logical<4> {
  %0 = fir.convert %arg0 : (i32) -> !fir.logical<4>
  return %0 : !fir.logical<4>
}
// -----
// CHECK-LABEL: @test_i32_l8
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i32
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i32
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i64
// CHECK: llvm.return [[t2]] : i64
func.func @test_i32_l8(%arg0: i32) -> !fir.logical<8> {
  %0 = fir.convert %arg0 : (i32) -> !fir.logical<8>
  return %0 : !fir.logical<8>
}
// -----
// CHECK-LABEL: @test_i64_l1
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i64
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i8
// CHECK: llvm.return [[t2]] : i8
func.func @test_i64_l1(%arg0: i64) -> !fir.logical<1> {
  %0 = fir.convert %arg0 : (i64) -> !fir.logical<1>
  return %0 : !fir.logical<1>
}
// -----
// CHECK-LABEL: @test_i64_l2
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i64
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i16
// CHECK: llvm.return [[t2]] : i16
func.func @test_i64_l2(%arg0: i64) -> !fir.logical<2> {
  %0 = fir.convert %arg0 : (i64) -> !fir.logical<2>
  return %0 : !fir.logical<2>
}
// -----
// CHECK-LABEL: @test_i64_l4
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i64
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i32
// CHECK: llvm.return [[t2]] : i32
func.func @test_i64_l4(%arg0: i64) -> !fir.logical<4> {
  %0 = fir.convert %arg0 : (i64) -> !fir.logical<4>
  return %0 : !fir.logical<4>
}
// -----
// CHECK-LABEL: @test_i64_l8
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i64
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i64
// CHECK: llvm.return [[t2]] : i64
func.func @test_i64_l8(%arg0: i64) -> !fir.logical<8> {
  %0 = fir.convert %arg0 : (i64) -> !fir.logical<8>
  return %0 : !fir.logical<8>
}
// -----
// CHECK-LABEL: @test_l1_l2
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i8
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i8
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i16
// CHECK: llvm.return [[t2]] : i16
func.func @test_l1_l2(%arg0: !fir.logical<1>) -> !fir.logical<2> {
  %0 = fir.convert %arg0 : (!fir.logical<1>) -> !fir.logical<2>
  return %0 : !fir.logical<2>
}
// -----
// CHECK-LABEL: @test_l1_l4
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i8
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i8
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i32
// CHECK: llvm.return [[t2]] : i32
func.func @test_l1_l4(%arg0: !fir.logical<1>) -> !fir.logical<4> {
  %0 = fir.convert %arg0 : (!fir.logical<1>) -> !fir.logical<4>
  return %0 : !fir.logical<4>
}
// -----
// CHECK-LABEL: @test_l1_l8
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i8
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i8
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i64
// CHECK: llvm.return [[t2]] : i64
func.func @test_l1_l8(%arg0: !fir.logical<1>) -> !fir.logical<8> {
  %0 = fir.convert %arg0 : (!fir.logical<1>) -> !fir.logical<8>
  return %0 : !fir.logical<8>
}
// -----
// CHECK-LABEL: @test_l2_l1
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i16
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i16
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i8
// CHECK: llvm.return [[t2]] : i8
func.func @test_l2_l1(%arg0: !fir.logical<2>) -> !fir.logical<1> {
  %0 = fir.convert %arg0 : (!fir.logical<2>) -> !fir.logical<1>
  return %0 : !fir.logical<1>
}
// -----
// CHECK-LABEL: @test_l2_l4
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i16
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i16
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i32
// CHECK: llvm.return [[t2]] : i32
func.func @test_l2_l4(%arg0: !fir.logical<2>) -> !fir.logical<4> {
  %0 = fir.convert %arg0 : (!fir.logical<2>) -> !fir.logical<4>
  return %0 : !fir.logical<4>
}
// -----
// CHECK-LABEL: @test_l2_l8
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i16
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i16
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i64
// CHECK: llvm.return [[t2]] : i64
func.func @test_l2_l8(%arg0: !fir.logical<2>) -> !fir.logical<8> {
  %0 = fir.convert %arg0 : (!fir.logical<2>) -> !fir.logical<8>
  return %0 : !fir.logical<8>
}
// -----
// CHECK-LABEL: @test_l4_l1
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i32
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i32
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i8
// CHECK: llvm.return [[t2]] : i8
func.func @test_l4_l1(%arg0: !fir.logical<4>) -> !fir.logical<1> {
  %0 = fir.convert %arg0 : (!fir.logical<4>) -> !fir.logical<1>
  return %0 : !fir.logical<1>
}
// -----
// CHECK-LABEL: @test_l4_l2
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i32
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i32
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i16
// CHECK: llvm.return [[t2]] : i16
func.func @test_l4_l2(%arg0: !fir.logical<4>) -> !fir.logical<2> {
  %0 = fir.convert %arg0 : (!fir.logical<4>) -> !fir.logical<2>
  return %0 : !fir.logical<2>
}
// -----
// CHECK-LABEL: @test_l4_l8
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i32
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i32
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i64
// CHECK: llvm.return [[t2]] : i64
func.func @test_l4_l8(%arg0: !fir.logical<4>) -> !fir.logical<8> {
  %0 = fir.convert %arg0 : (!fir.logical<4>) -> !fir.logical<8>
  return %0 : !fir.logical<8>
}
// -----
// CHECK-LABEL: @test_l8_l1
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i64
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i8
// CHECK: llvm.return [[t2]] : i8
func.func @test_l8_l1(%arg0: !fir.logical<8>) -> !fir.logical<1> {
  %0 = fir.convert %arg0 : (!fir.logical<8>) -> !fir.logical<1>
  return %0 : !fir.logical<1>
}
// -----
// CHECK-LABEL: @test_l8_l2
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i64
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i16
// CHECK: llvm.return [[t2]] : i16
func.func @test_l8_l2(%arg0: !fir.logical<8>) -> !fir.logical<2> {
  %0 = fir.convert %arg0 : (!fir.logical<8>) -> !fir.logical<2>
  return %0 : !fir.logical<2>
}
// -----
// CHECK-LABEL: @test_l8_l4
// CHECK: [[t0:%[0-9]*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: [[t1:%[0-9]*]] = llvm.icmp "ne" %arg0, [[t0]] : i64
// CHECK: [[t2:%[0-9]*]] = llvm.zext [[t1]] : i1 to i32
// CHECK: llvm.return [[t2]] : i32
func.func @test_l8_l4(%arg0: !fir.logical<8>) -> !fir.logical<4> {
  %0 = fir.convert %arg0 : (!fir.logical<8>) -> !fir.logical<4>
  return %0 : !fir.logical<4>
}
