add_custom_target(libc-fputil-tests)

if(NOT LIBC_TARGET_ARCHITECTURE_IS_GPU)
  add_fp_unittest(
    dyadic_float_test
    NEED_MPFR
    SUITE
      libc-fputil-tests
    SRCS
      dyadic_float_test.cpp
    DEPENDS
      libc.src.__support.FPUtil.dyadic_float
  )
endif()

add_libc_test(
  fpbits_test
  SUITE
    libc-fputil-tests
  SRCS
    fpbits_test.cpp
  DEPENDS
    libc.src.__support.FPUtil.fp_bits
    libc.src.__support.FPUtil.fpbits_str
)

if(NOT LIBC_TARGET_ARCHITECTURE_IS_GPU)
  add_fp_unittest(
    rounding_mode_test
    SUITE
      libc-fputil-tests
    SRCS
      rounding_mode_test.cpp
    DEPENDS
      libc.src.__support.FPUtil.rounding_mode
  )
endif()
