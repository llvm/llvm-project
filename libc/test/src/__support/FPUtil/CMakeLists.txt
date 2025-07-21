add_custom_target(libc-fputil-tests)

add_fp_unittest(
  dyadic_float_test
  NEED_MPFR
  SUITE
    libc-fputil-tests
  SRCS
    dyadic_float_test.cpp
  DEPENDS
    libc.src.__support.FPUtil.dyadic_float
    libc.src.__support.macros.properties.types
  COMPILE_OPTIONS
    # Prevent constant folding with a default rounding mode.
    "-frounding-math"
)

add_libc_test(
  fpbits_test
  SUITE
    libc-fputil-tests
  SRCS
    fpbits_test.cpp
  DEPENDS
    libc.src.__support.FPUtil.fp_bits
    libc.src.__support.FPUtil.fpbits_str
    libc.src.__support.integer_literals
    libc.src.__support.macros.properties.types
    libc.src.__support.sign
)

add_fp_unittest(
  rounding_mode_test
  SUITE
    libc-fputil-tests
  SRCS
    rounding_mode_test.cpp
  DEPENDS
    libc.src.__support.FPUtil.rounding_mode
)

# TODO: Temporally disable bfloat16 test until MPCommon target is updated
# https://github.com/llvm/llvm-project/pull/149678
if(LLVM_LIBC_FULL_BUILD)
  return()
endif()

add_fp_unittest(
  bfloat16_test
  NEED_MPFR
  SUITE
    libc-fputil-tests
  SRCS
    bfloat16_test.cpp
  DEPENDS
    libc.src.__support.FPUtil.bfloat16
)
