// RUN: fir-opt --target-rewrite="target=mips64-unknown-linux-gnuabi64" %s | FileCheck %s
// RUN: fir-opt --target-rewrite="target=mips64el-unknown-linux-gnuabi64" %s | FileCheck %s
// RUN: fir-opt --target-rewrite="target=mips64-unknown-linux-gnuabin32" %s | FileCheck %s
// RUN: fir-opt --target-rewrite="target=mips64el-unknown-linux-gnuabin32" %s | FileCheck %s

// Single and double precision complex results use two floating-point registers.
// CHECK-LABEL: func.func @return4() -> tuple<f32, f32>
func.func @return4() -> complex<f32> {
  %z = fir.undefined complex<f32>
  return %z : complex<f32>
}
// CHECK-LABEL: func.func @return8() -> tuple<f64, f64>
func.func @return8() -> complex<f64> {
  %z = fir.undefined complex<f64>
  return %z : complex<f64>
}
// Quad precision results use a hidden pointer.
// CHECK-LABEL: func.func @return16(
// CHECK-SAME: !fir.ref<tuple<f128, f128>> {llvm.align = 16 : i32, llvm.sret = tuple<f128, f128>})
func.func @return16() -> complex<f128> {
  %z = fir.undefined complex<f128>
  return %z : complex<f128>
}

// 32-bit integers, including unsigned integers, must be sign extended.
// CHECK: func.func private @integers(i1 {llvm.zeroext}, i8 {llvm.signext}, ui8 {llvm.zeroext}, i16 {llvm.signext}, ui16 {llvm.zeroext}, i32 {llvm.signext}, ui32 {llvm.signext}, i64) -> (i32 {llvm.signext})
func.func private @integers(i1, i8, ui8, i16, ui16, i32, ui32, i64) -> i32 attributes {fir.runtime}
// CHECK: func.func private @unsigned_result() -> (ui32 {llvm.signext})
func.func private @unsigned_result() -> ui32 attributes {fir.runtime}

// CHECK-LABEL: func.func @complex32_after0({{%[^:]+}}: f32, {{%[^:]+}}: f32, {{%[^:]+}}: i64) -> tuple<f32, f32>
func.func @complex32_after0(%a0: complex<f32>, %a1: i64) -> complex<f32> {
  // CHECK: fir.call @complex32_after0({{.*}}) : (f32, f32, i64) -> tuple<f32, f32>
  %r = fir.call @complex32_after0(%a0, %a1) : (complex<f32>, i64) -> complex<f32>
  return %r : complex<f32>
}

// CHECK-LABEL: func.func @complex32_after6({{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: f32, {{%[^:]+}}: f32, {{%[^:]+}}: i64) -> tuple<f32, f32>
func.func @complex32_after6(%a0: i64, %a1: i64, %a2: i64, %a3: i64, %a4: i64, %a5: i64, %a6: complex<f32>, %a7: i64) -> complex<f32> {
  // CHECK: fir.call @complex32_after6({{.*}}) : (i64, i64, i64, i64, i64, i64, f32, f32, i64) -> tuple<f32, f32>
  %r = fir.call @complex32_after6(%a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7) : (i64, i64, i64, i64, i64, i64, complex<f32>, i64) -> complex<f32>
  return %r : complex<f32>
}

// CHECK-LABEL: func.func @complex32_after7({{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64) -> tuple<f32, f32>
func.func @complex32_after7(%a0: i64, %a1: i64, %a2: i64, %a3: i64, %a4: i64, %a5: i64, %a6: i64, %a7: complex<f32>, %a8: i64) -> complex<f32> {
  // CHECK: fir.call @complex32_after7({{.*}}) : (i64, i64, i64, i64, i64, i64, i64, i64, i64) -> tuple<f32, f32>
  %r = fir.call @complex32_after7(%a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7, %a8) : (i64, i64, i64, i64, i64, i64, i64, complex<f32>, i64) -> complex<f32>
  return %r : complex<f32>
}

// CHECK-LABEL: func.func @complex32_after8({{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64) -> tuple<f32, f32>
func.func @complex32_after8(%a0: i64, %a1: i64, %a2: i64, %a3: i64, %a4: i64, %a5: i64, %a6: i64, %a7: i64, %a8: complex<f32>, %a9: i64) -> complex<f32> {
  // CHECK: fir.call @complex32_after8({{.*}}) : (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> tuple<f32, f32>
  %r = fir.call @complex32_after8(%a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7, %a8, %a9) : (i64, i64, i64, i64, i64, i64, i64, i64, complex<f32>, i64) -> complex<f32>
  return %r : complex<f32>
}

// CHECK-LABEL: func.func @complex64_after0({{%[^:]+}}: f64, {{%[^:]+}}: f64, {{%[^:]+}}: i64) -> tuple<f64, f64>
func.func @complex64_after0(%a0: complex<f64>, %a1: i64) -> complex<f64> {
  // CHECK: fir.call @complex64_after0({{.*}}) : (f64, f64, i64) -> tuple<f64, f64>
  %r = fir.call @complex64_after0(%a0, %a1) : (complex<f64>, i64) -> complex<f64>
  return %r : complex<f64>
}

// CHECK-LABEL: func.func @complex64_after6({{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: f64, {{%[^:]+}}: f64, {{%[^:]+}}: i64) -> tuple<f64, f64>
func.func @complex64_after6(%a0: i64, %a1: i64, %a2: i64, %a3: i64, %a4: i64, %a5: i64, %a6: complex<f64>, %a7: i64) -> complex<f64> {
  // CHECK: fir.call @complex64_after6({{.*}}) : (i64, i64, i64, i64, i64, i64, f64, f64, i64) -> tuple<f64, f64>
  %r = fir.call @complex64_after6(%a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7) : (i64, i64, i64, i64, i64, i64, complex<f64>, i64) -> complex<f64>
  return %r : complex<f64>
}

// CHECK-LABEL: func.func @complex64_after7({{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: tuple<i64, i64>, {{%[^:]+}}: i64) -> tuple<f64, f64>
func.func @complex64_after7(%a0: i64, %a1: i64, %a2: i64, %a3: i64, %a4: i64, %a5: i64, %a6: i64, %a7: complex<f64>, %a8: i64) -> complex<f64> {
  // CHECK: fir.call @complex64_after7({{.*}}) : (i64, i64, i64, i64, i64, i64, i64, tuple<i64, i64>, i64) -> tuple<f64, f64>
  %r = fir.call @complex64_after7(%a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7, %a8) : (i64, i64, i64, i64, i64, i64, i64, complex<f64>, i64) -> complex<f64>
  return %r : complex<f64>
}

// CHECK-LABEL: func.func @complex64_after8({{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: i64, {{%[^:]+}}: tuple<i64, i64>, {{%[^:]+}}: i64) -> tuple<f64, f64>
func.func @complex64_after8(%a0: i64, %a1: i64, %a2: i64, %a3: i64, %a4: i64, %a5: i64, %a6: i64, %a7: i64, %a8: complex<f64>, %a9: i64) -> complex<f64> {
  // CHECK: fir.call @complex64_after8({{.*}}) : (i64, i64, i64, i64, i64, i64, i64, i64, tuple<i64, i64>, i64) -> tuple<f64, f64>
  %r = fir.call @complex64_after8(%a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7, %a8, %a9) : (i64, i64, i64, i64, i64, i64, i64, i64, complex<f64>, i64) -> complex<f64>
  return %r : complex<f64>
}

// Previously split complex arguments each consume two slots, even for f32.
// CHECK: func.func private @five_complex(f32, f32, f32, f32, f32, f32, f32, f32, i64)
func.func private @five_complex(complex<f32>, complex<f32>, complex<f32>, complex<f32>, complex<f32>)

// Count the hidden result pointer as an argument slot.
// CHECK-LABEL: func.func private @hidden_result(
// CHECK-SAME: !fir.ref<tuple<f128, f128>> {llvm.align = 16 : i32, llvm.sret = tuple<f128, f128>}, i64, i64, i64, i64, i64, i64, i64)
func.func private @hidden_result(i64, i64, i64, i64, i64, i64, complex<f32>) -> complex<f128>

// Function addresses and indirect calls must use the same rewritten signature.
// CHECK-LABEL: func.func @indirect(
func.func @indirect(%a: i64, %z: complex<f32>) -> complex<f32> {
  // CHECK: fir.address_of(@complex32_after7) : (i64, i64, i64, i64, i64, i64, i64, i64, i64) -> tuple<f32, f32>
  %fn = fir.address_of(@complex32_after7) : (i64, i64, i64, i64, i64, i64, i64, complex<f32>, i64) -> complex<f32>
  // CHECK: fir.call %{{.*}}({{.*}}) : (i64, i64, i64, i64, i64, i64, i64, i64, i64) -> tuple<f32, f32>
  %r = fir.call %fn(%a, %a, %a, %a, %a, %a, %a, %z, %a) : (i64, i64, i64, i64, i64, i64, i64, complex<f32>, i64) -> complex<f32>
  return %r : complex<f32>
}
