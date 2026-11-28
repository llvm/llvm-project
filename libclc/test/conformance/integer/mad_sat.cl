// RUN: %libclc-compile
// RUN: %libclc-run --kernel mad_sat_i64 --threads-x 1 %t
// RUN: %libclc-run --kernel mad_sat_i64_vec --threads-x 1 %t

#include "conformance.h"

// OpenCL C v3.0.19, Sec. 6.15.3: mad_sat() saturates a * b + c.
__kernel void mad_sat_i64(void) {
  CHECK_EQ(
      mad_sat(TEST_INPUT(long, 3L), TEST_INPUT(long, 4L), TEST_INPUT(long, 5L)),
      17L);
  CHECK_EQ(mad_sat(TEST_INPUT(long, 0L), TEST_INPUT(long, -5L),
                   TEST_INPUT(long, 7L)),
           7L);
  CHECK_EQ(mad_sat(TEST_INPUT(long, LONG_MAX), TEST_INPUT(long, 2L),
                   TEST_INPUT(long, LONG_MIN)),
           LONG_MAX - 1);
  CHECK_EQ(mad_sat(TEST_INPUT(long, LONG_MAX), TEST_INPUT(long, 2L),
                   TEST_INPUT(long, 0L)),
           LONG_MAX);
  CHECK_EQ(mad_sat(TEST_INPUT(long, LONG_MIN), TEST_INPUT(long, 2L),
                   TEST_INPUT(long, 0L)),
           LONG_MIN);
  CHECK_EQ(mad_sat(TEST_INPUT(long, 1L), TEST_INPUT(long, 1L),
                   TEST_INPUT(long, LONG_MAX)),
           LONG_MAX);
  CHECK_EQ(mad_sat(TEST_INPUT(long, -1L), TEST_INPUT(long, 1L),
                   TEST_INPUT(long, LONG_MIN)),
           LONG_MIN);
  CHECK_EQ(mad_sat(TEST_INPUT(long, LONG_MIN), TEST_INPUT(long, -1L),
                   TEST_INPUT(long, -1L)),
           LONG_MAX);
  CHECK_EQ(mad_sat(TEST_INPUT(long, LONG_MAX), TEST_INPUT(long, 1L),
                   TEST_INPUT(long, LONG_MIN)),
           -1L);
  CHECK_EQ(mad_sat(TEST_INPUT(long, LONG_MIN), TEST_INPUT(long, 1L),
                   TEST_INPUT(long, 5L)),
           LONG_MIN + 5);
  CHECK_EQ(mad_sat(TEST_INPUT(long, LONG_MAX), TEST_INPUT(long, LONG_MAX),
                   TEST_INPUT(long, 0L)),
           LONG_MAX);
  CHECK_EQ(mad_sat(TEST_INPUT(long, LONG_MIN), TEST_INPUT(long, LONG_MAX),
                   TEST_INPUT(long, 0L)),
           LONG_MIN);
}

// Every lane carries into the high half.
__kernel void mad_sat_i64_vec(void) {
  long4 x = TEST_INPUT(long4, (long4)(-1L, LONG_MIN, -1L, -2L));
  long4 y = TEST_INPUT(long4, (long4)(1L, -1L, -1L, 3L));
  long4 z = TEST_INPUT(long4, (long4)(LONG_MIN, -1L, -1L, 7L));
  CHECK(all(mad_sat(x, y, z) == (long4)(LONG_MIN, LONG_MAX, 0L, 1L)));
}
