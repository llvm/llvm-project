// RUN: not %clang_dxc -enable-16bit-types -T cs_6_0 -HV 202x %s 2>&1  | FileCheck %s

// check we error on 16 bit type if shader model is too old
// CHECK: '-enable-16bit-types' option requires target HLSL Version >= 2018 and shader model >= 6.2, but HLSL Version is 'hlsl202x' and shader model is '6.0'
uint16_t test_uint16_t_error(uint16_t p0, int p1) {
  return clamp(p0, p0, p1);
}
