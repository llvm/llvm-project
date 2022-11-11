add_fp_unittest(
  ceil_test
  NEED_MPFR
  SUITE
    libc_math_unittests
  SRCS
    ../ceil_test.cpp
  DEPENDS
    libc.include.math
    libc.src.math.generic.ceil
)

add_fp_unittest(
  ceilf_test
  NEED_MPFR
  SUITE
    libc_math_unittests
  SRCS
    ../ceilf_test.cpp
  DEPENDS
    libc.include.math
    libc.src.math.generic.ceilf
)

add_fp_unittest(
  ceill_test
  NEED_MPFR
  SUITE
    libc_math_unittests
  SRCS
    ../ceill_test.cpp
  DEPENDS
    libc.include.math
    libc.src.math.generic.ceill
)

