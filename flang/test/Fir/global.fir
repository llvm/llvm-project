// RUN: tco %s | FileCheck %s

// CHECK: @g_i0 = global i32 0
fir.global @g_i0 : i32 {
  %1 = constant 0 : i32
  fir.has_value %1 : i32
}

// CHECK: @g_i2 = global i32 2
fir.global @g_i2 : i32 {
  %1 = constant 2 : i32
  fir.has_value %1 : i32
}

// CHECK: @g_ci5 = constant i32 5
fir.global @g_ci5 constant : i32 {
  %c = constant 5 : i32
  fir.has_value %c : i32
}

// CHECK: @i_i515 = internal global i32 515
fir.global internal @i_i515 (515:i32) : i32

// CHECK: @C_i511 = common global i32 511
fir.global common @C_i511 (511:i32) : i32

// CHECK: @w_i86 = weak global i32 86
fir.global weak @w_i86 (86:i32) : i32

// CHECK: @str1 = global [6 x i8] c"Hello!"
fir.global @str1 : !fir.array<6 x !fir.char<1>> {
  %1 = fir.string_lit "Hello!"(6) : !fir.char<1>
  fir.has_value %1 : !fir.array<6 x !fir.char<1>>
}
