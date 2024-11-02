// RUN: tco %s | FileCheck %s

// CHECK-LABEL: @character_literal1
func.func @character_literal1() -> !fir.char<1,13> {
  %0 = fir.string_lit "Hello, World!"(13) : !fir.char<1,13>
  // CHECK: ret [13 x i8] c"Hello, World!"
  return %0 : !fir.char<1,13>
}

// CHECK-LABEL: @character_literal2
func.func @character_literal2() -> !fir.char<2,2> {
  %0 = fir.string_lit [234, 456](2) : !fir.char<2,2>
  // CHECK: ret [2 x i16] [i16 234, i16 456]
  return %0 : !fir.char<2,2>
}

// CHECK-LABEL: @character_literal4
func.func @character_literal4() -> !fir.char<4,3> {
  %0 = fir.string_lit [89123, 999256, 4](3) : !fir.char<4,3>
  // CHECK: ret [3 x i32] [i32 89123, i32 999256, i32 4]
  return %0 : !fir.char<4,3>
}
