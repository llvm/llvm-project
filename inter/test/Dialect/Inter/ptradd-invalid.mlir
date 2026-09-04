// RUN: inter-opt --split-input-file -verify-diagnostics %s

func.func @different_cardinalities(
    %base: !xw.simd<!xw.ptr<#xw.global>, 8>,
    %offset: !xw.simd<i32, 16>) attributes {xw.simd_width = 32 : i64} {
  // expected-error@+1 {{base and offset SIMD cardinalities must match}}
  %ptr = xw.ptradd %base, %offset
      : !xw.simd<!xw.ptr<#xw.global>, 8>, !xw.simd<i32, 16>
      -> !xw.simd<!xw.ptr<#xw.global>, 16>
  return
}

// -----

func.func @not_a_pointer(%base: i64, %offset: i64)
    attributes {xw.simd_width = 32 : i64} {
  // expected-error@+1 {{expected an XW pointer or SIMD of XW pointers}}
  %ptr = xw.ptradd %base, %offset : i64, i64 -> i64
  return
}
