//===----------------------------------------------------------------------===//
//
// Shared helpers for libclc execution conformance tests. These kernels are
// launched on-device and report failure by trapping.
//
//===----------------------------------------------------------------------===//

#ifndef LIBCLC_TEST_CONFORMANCE_H
#define LIBCLC_TEST_CONFORMANCE_H

// Yields an input value the compiler cannot optimize out.
#define TEST_INPUT(TYPE, VALUE)                                                \
  ({                                                                           \
    volatile TYPE __clc_in = (VALUE);                                          \
    __clc_in;                                                                  \
  })

// Check the condition and submit a hardware trap on failure.
#define CHECK(COND)                                                            \
  do {                                                                         \
    if (!(COND))                                                               \
      __builtin_verbose_trap("libclc", "check failed: " #COND);                \
  } while (0)

#define CHECK_EQ(LHS, RHS) CHECK((LHS) == (RHS))
#define CHECK_NE(LHS, RHS) CHECK((LHS) != (RHS))
#define CHECK_LT(LHS, RHS) CHECK((LHS) < (RHS))
#define CHECK_LE(LHS, RHS) CHECK((LHS) <= (RHS))
#define CHECK_GT(LHS, RHS) CHECK((LHS) > (RHS))
#define CHECK_GE(LHS, RHS) CHECK((LHS) >= (RHS))

// Representable values between X and Y. Does not handle fractional ULP values
// and rounds up instead, not fully accurate for all cases.
static inline uint __clc_test_ulp_f32(float x, float y) {
  uint a = as_uint(x), b = as_uint(y);
  a = (a >> 31) ? 0x80000000U - (a & 0x7fffffffU) : 0x80000000U + a;
  b = (b >> 31) ? 0x80000000U - (b & 0x7fffffffU) : 0x80000000U + b;
  return a > b ? a - b : b - a;
}

// The reference must be of the result's own type, or it converts and the
// distance is measured on the wrong grid.
#define CHECK_ULP_F32(GOT, REF, ULPS)                                          \
  do {                                                                         \
    _Static_assert(__builtin_types_compatible_p(__typeof__(GOT), float) &&     \
                       __builtin_types_compatible_p(__typeof__(REF), float),   \
                   "CHECK_ULP_F32 needs float operands");                      \
    CHECK(__clc_test_ulp_f32((GOT), (REF)) <= (ULPS));                         \
  } while (0)

#ifdef __opencl_c_fp64
static inline ulong __clc_test_ulp_f64(double x, double y) {
  ulong a = as_ulong(x), b = as_ulong(y);
  a = (a >> 63) ? 0x8000000000000000UL - (a & 0x7fffffffffffffffUL)
                : 0x8000000000000000UL + a;
  b = (b >> 63) ? 0x8000000000000000UL - (b & 0x7fffffffffffffffUL)
                : 0x8000000000000000UL + b;
  return a > b ? a - b : b - a;
}

#define CHECK_ULP_F64(GOT, REF, ULPS)                                          \
  do {                                                                         \
    _Static_assert(__builtin_types_compatible_p(__typeof__(GOT), double) &&    \
                       __builtin_types_compatible_p(__typeof__(REF), double),  \
                   "CHECK_ULP_F64 needs double operands");                     \
    CHECK(__clc_test_ulp_f64((GOT), (REF)) <= (ULPS));                         \
  } while (0)
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

static inline ushort __clc_test_ulp_f16(half x, half y) {
  ushort a = as_ushort(x), b = as_ushort(y);
  a = (a >> 15) ? 0x8000U - (a & 0x7fffU) : 0x8000U + a;
  b = (b >> 15) ? 0x8000U - (b & 0x7fffU) : 0x8000U + b;
  return a > b ? a - b : b - a;
}

#define CHECK_ULP_F16(GOT, REF, ULPS)                                          \
  do {                                                                         \
    _Static_assert(__builtin_types_compatible_p(__typeof__(GOT), half) &&      \
                       __builtin_types_compatible_p(__typeof__(REF), half),    \
                   "CHECK_ULP_F16 needs half operands");                       \
    CHECK(__clc_test_ulp_f16((GOT), (REF)) <= (ULPS));                         \
  } while (0)
#endif

#endif // LIBCLC_TEST_CONFORMANCE_H
