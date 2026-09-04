// RUN: inter-opt %s | FileCheck %s

func.func @ptradd(%base: !xw.ptr<#xw.global>, %uniform: i64,
                  %lane: !xw.simd<i32, 8>)
    attributes {xw.simd_width = 32 : i64} {
  // CHECK: xw.ptradd {{%.*}}, {{%.*}} : !xw.ptr<#xw.global>, i64 -> !xw.ptr<#xw.global>
  %uniform_ptr = xw.ptradd %base, %uniform
      : !xw.ptr<#xw.global>, i64 -> !xw.ptr<#xw.global>
  // CHECK: xw.ptradd {{%.*}}, {{%.*}} : !xw.ptr<#xw.global>, !xw.simd<i32, 8> -> !xw.simd<!xw.ptr<#xw.global>, 8>
  %lane_ptr = xw.ptradd %base, %lane
      : !xw.ptr<#xw.global>, !xw.simd<i32, 8>
      -> !xw.simd<!xw.ptr<#xw.global>, 8>
  return
}
