add_libc_testsuite(libc_fputil_unittests)

add_fp_unittest(
  dyadic_float_test
  NEED_MPFR
  SUITE
    libc_fputil_unittests
  SRCS
    dyadic_float_test.cpp
  DEPENDS
    libc.src.__support.FPUtil.dyadic_float
)
