// RUN: not %clang_dxc -enable-16bit-types -T cs_6_0 -HV 202x %s 2>&1  | FileCheck %s

// check we error on 16 bit type if shader model is too old
// CHECK: '-enable-16bit-types' option requires target HLSL Version >= 2018 and shader model >= 6.2, but HLSL Version is 'hlsl202x' and shader model is '6.0'
int16_t test_int16_t_error(int16_t p0, int p1) {
  return clamp(p0, p0, p1);
}

int16_t3 test_int16_t3_error(int16_t3 p0, int3 p1) {
  return clamp(p0, p0, p1);
}

half test_half_error(half p0, int p1) {
  return clamp(p0, p1, p1);
}

half3 test_half3_error(half3 p0, int3 p1) {
  return clamp(p0, p0, p1);
}

uint16_t test_uint16_t_error(uint16_t p0, int p1) {
  return clamp(p0, p0, p1);
}

uint16_t3 test_uint16_t3_error(uint16_t3 p0, int3 p1) {
  return clamp(p0, p1, p1);
}
