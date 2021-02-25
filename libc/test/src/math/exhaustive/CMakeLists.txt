add_libc_exhaustive_testsuite(libc_math_exhaustive_tests)

add_fp_unittest(
  sqrtf_test
  NEED_MPFR
  SUITE
    libc_math_exhaustive_tests
  SRCS
    sqrtf_test.cpp
  DEPENDS
    libc.include.math
    libc.src.math.sqrtf
    libc.utils.FPUtil.fputil
)
