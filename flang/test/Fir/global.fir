// RUN: tco %s | FileCheck %s
// RUN: %flang_fc1 -emit-llvm  %s -o - | FileCheck %s

// CHECK: @g_i0 = global i32 0
fir.global @g_i0 : i32 {
  %1 = arith.constant 0 : i32
  fir.has_value %1 : i32
}

// CHECK: @g_i2 = global i32 2
fir.global @g_i2 : i32 {
  %1 = arith.constant 2 : i32
  fir.has_value %1 : i32
}

// CHECK: @g_ci5 = constant i32 5
fir.global @g_ci5 constant : i32 {
  %c = arith.constant 5 : i32
  fir.has_value %c : i32
}

// CHECK: @i_i515 = internal global i32 515
fir.global internal @i_i515 (515:i32) : i32

// CHECK: @C_i511 = common global i32 0
fir.global common @C_i511 (0:i32) : i32

// CHECK: @w_i86 = weak global i32 86
fir.global weak @w_i86 (86:i32) : i32

// CHECK: @str1 = global [6 x i8] c"Hello!"
fir.global @str1 : !fir.char<1,6> {
  %1 = fir.string_lit "Hello!"(6) : !fir.char<1,6>
  fir.has_value %1 : !fir.char<1,6>
}

// CHECK: @_QEmask = internal global [32 x i32] [i32 1, i32 1
fir.global internal @_QEmask : !fir.array<32xi32> {
  %c0_i32 = arith.constant 1 : i32
  %0 = fir.undefined !fir.array<32xi32>
  %2 = fir.insert_on_range %0, %c0_i32 from (0) to (31) : (!fir.array<32xi32>, i32) -> !fir.array<32xi32>
  fir.has_value %2 : !fir.array<32xi32>
}

// CHECK: @_QEmultiarray = internal global [32 x [32 x i32]]
fir.global internal @_QEmultiarray : !fir.array<32x32xi32> {
  %c0_i32 = arith.constant 1 : i32
  %0 = fir.undefined !fir.array<32x32xi32>
  %2 = fir.insert_on_range %0, %c0_i32 from (0,0) to (31,31) : (!fir.array<32x32xi32>, i32) -> !fir.array<32x32xi32>
  fir.has_value %2 : !fir.array<32x32xi32>
}

// CHECK: @_QEmasklogical = internal global [32768 x i32] [i32 -1, i32 -1,
fir.global internal @_QEmasklogical : !fir.array<32768x!fir.logical<4>> {
  %true = arith.constant true
  %0 = fir.undefined !fir.array<32768x!fir.logical<4>>
  %1 = fir.convert %true : (i1) -> !fir.logical<4>
  %2 = fir.insert_on_range %0, %1 from (0) to (32767) : (!fir.array<32768x!fir.logical<4>>, !fir.logical<4>) -> !fir.array<32768x!fir.logical<4>>
  fir.has_value %2 : !fir.array<32768x!fir.logical<4>>
}
