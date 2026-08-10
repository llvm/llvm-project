if(NOT LIBC_COMPILER_HAS_FIXED_POINT)
  return()
endif()

add_custom_target(libc-fixed-point-tests)

add_libc_test(
  fx_bits_test
  SUITE
    libc-fixed-point-tests
  SRCS
    fx_bits_test.cpp
  DEPENDS
    libc.src.__support.fixed_point.fx_bits
    libc.src.__support.integer_literals
)
