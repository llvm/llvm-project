// Test no-struct-conversion of target-rewrite pass.
// RUN: fir-opt -target-rewrite="no-struct-conversion" %s | FileCheck %s

func.func @test(%arg0: !fir.type<t{i:i32}>) {
  return
}

func.func @test_call(%arg0: !fir.type<t{i:i32}>) {
  fir.call @test(%arg0) : (!fir.type<t{i:i32}>) -> ()
  return
}

func.func @test_addr_off() {
  %0 = fir.address_of(@test) : (!fir.type<t{i:i32}>) -> ()
  return
}

// CHECK-LABEL:  func.func @test(%{{.*}}: !fir.type<t{i:i32}>) {

// CHECK-LABEL:  func.func @test_call(
// CHECK-SAME:                        %[[ARG0:.*]]: !fir.type<t{i:i32}>) {
// CHECK:    fir.call @test(%[[ARG0]]) : (!fir.type<t{i:i32}>) -> ()

// CHECK-LABEL:  func.func @test_addr_off() {
// CHECK:    fir.address_of(@test) : (!fir.type<t{i:i32}>) -> ()
