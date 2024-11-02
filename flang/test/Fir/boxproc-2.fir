// Test fir.boxproc type conversion in the boxed-procedure pass.
// RUN: fir-opt --boxed-procedure %s | FileCheck %s

//CHECK-LABEL:  func.func private @test1(!fir.type<a{x:i32,y:f64}>, () -> ()) -> none
func.func private @test1(!fir.type<a{x:i32, y:f64}>, !fir.boxproc<() -> ()>) -> none

//CHECK-LABEL:  func.func private @test2((!fir.type<a{x:i32,y:f64}>) -> ()) -> none
func.func private @test2(!fir.boxproc<(!fir.type<a{x:i32, y:f64}>) -> ()>) -> none

//CHECK-LABEL:  func.func private @test3(() -> !fir.type<a{x:i32,y:f64}>) -> none
func.func private @test3(!fir.boxproc<() -> (!fir.type<a{x:i32, y:f64}>)>) -> none

//CHECK-LABEL:  func.func private @test5(((i32) -> f32) -> ())
func.func private @test5(!fir.boxproc<(!fir.boxproc<(i32) -> (f32)>) -> ()>)
