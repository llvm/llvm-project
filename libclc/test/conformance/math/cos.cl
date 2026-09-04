// RUN: %libclc-compile
// RUN: %libclc-run --kernel cos_f32 --threads-x 1 %t
// RUN: %libclc-run --kernel cos_f64 --threads-x 1 %t
// RUN: %libclc-run --kernel cos_f16 --threads-x 1 %t

#include "conformance.h"

// OpenCL C v3.0.19, Sec. 7.4, Table 65: cos() is allowed 4 ULP. The arguments
// cover the argument reduction paths.
__kernel void cos_f32(void) {
  CHECK_ULP_F32(cos(TEST_INPUT(float, 1.0f)), 0x1.14a28p-1f, 4);

  // Nearest float to pi/2, where the result all but cancels.
  CHECK_ULP_F32(cos(TEST_INPUT(float, 0x1.921fb6p+0f)), -0x1.777a5cp-25f, 4);

  // Nearest float to pi.
  CHECK_ULP_F32(cos(TEST_INPUT(float, 0x1.921fb6p+1f)), -0x1p+0f, 4);

  CHECK_ULP_F32(cos(TEST_INPUT(float, 100.0f)), 0x1.b981dcp-1f, 4);

  // Needs more bits of 2/pi than a double holds.
  CHECK_ULP_F32(cos(TEST_INPUT(float, 0x1.0p+64f)), -0x1.ffdb8p-1f, 4);
  CHECK_ULP_F32(cos(TEST_INPUT(float, 0x1.fffffep+127f)), 0x1.b4bf2cp-1f, 4);

  // Not correctly rounded, so the bound is exercised.
  CHECK_ULP_F32(cos(TEST_INPUT(float, 0x1.0c158cp+0f)), 0x1.fffe96p-2f, 4);

  CHECK_EQ(cos(TEST_INPUT(float, 0.0f)), 1.0f);
  CHECK_EQ(cos(TEST_INPUT(float, -0.0f)), 1.0f);
  CHECK(isnan(cos(TEST_INPUT(float, INFINITY))));
  CHECK(isnan(cos(TEST_INPUT(float, -INFINITY))));
  CHECK(isnan(cos(TEST_INPUT(float, NAN))));
}

// OpenCL C v3.0.19, Sec. 7.4, Table 68: cos() is allowed 4 ULP.
__kernel void cos_f64(void) {
#ifdef __opencl_c_fp64
  CHECK_ULP_F64(cos(TEST_INPUT(double, 1.0)), 0x1.14a280fb5068cp-1, 4);

  // Nearest double to pi/2, then to pi.
  CHECK_ULP_F64(cos(TEST_INPUT(double, 0x1.921fb54442d18p+0)),
                0x1.1a62633145c07p-54, 4);
  CHECK_ULP_F64(cos(TEST_INPUT(double, 0x1.921fb54442d18p+1)), -0x1p+0, 4);

  CHECK_ULP_F64(cos(TEST_INPUT(double, 100.0)), 0x1.b981dbf665fdfp-1, 4);

  CHECK_ULP_F64(cos(TEST_INPUT(double, 0x1.0p+64)), -0x1.ffdb7fa3fe34dp-1, 4);
  CHECK_ULP_F64(cos(TEST_INPUT(double, 0x1.fffffffffffffp+1023)),
                -0x1.fffe62ecfab75p-1, 4);

  // Not correctly rounded, so the bound is exercised.
  CHECK_ULP_F64(cos(TEST_INPUT(double, 0x1.0cf7ef9db22d1p+9)),
                -0x1.7f59309b88661p-1, 4);

  CHECK_EQ(cos(TEST_INPUT(double, 0.0)), 1.0);
  CHECK_EQ(cos(TEST_INPUT(double, -0.0)), 1.0);
  CHECK(isnan(cos(TEST_INPUT(double, (double)INFINITY))));
  CHECK(isnan(cos(TEST_INPUT(double, -(double)INFINITY))));
  CHECK(isnan(cos(TEST_INPUT(double, (double)NAN))));
#endif
}

// OpenCL C v3.0.19, Sec. 7.4, Table 69 (Full Profile): half cos() is allowed
// 2 ULP.
__kernel void cos_f16(void) {
#ifdef cl_khr_fp16
  CHECK_ULP_F16(cos(TEST_INPUT(half, 1.0h)), 0x1.14cp-1h, 2);

  // Nearest half to pi/2, then to pi.
  CHECK_ULP_F16(cos(TEST_INPUT(half, 0x1.92p+0h)), 0x1.fb4p-12h, 2);
  CHECK_ULP_F16(cos(TEST_INPUT(half, 0x1.92p+1h)), -0x1p+0h, 2);

  // Widest argument needing reduction.
  CHECK_ULP_F16(cos(TEST_INPUT(half, 0x1.ffcp+15h)), -0x1.c3cp-3h, 2);

  // Not correctly rounded, so the bound is exercised.
  CHECK_ULP_F16(cos(TEST_INPUT(half, 0x1.8acp-4h)), 0x1.fd8p-1h, 2);

  // Smallest subnormal, where the result rounds to one.
  CHECK_ULP_F16(cos(TEST_INPUT(half, 0x1.0p-24h)), 0x1p+0h, 2);

  CHECK_EQ(cos(TEST_INPUT(half, 0.0h)), 1.0h);
  CHECK_EQ(cos(TEST_INPUT(half, -0.0h)), 1.0h);
  CHECK(isnan(cos(TEST_INPUT(half, (half)INFINITY))));
  CHECK(isnan(cos(TEST_INPUT(half, (half)-INFINITY))));
  CHECK(isnan(cos(TEST_INPUT(half, (half)NAN))));
#endif
}
