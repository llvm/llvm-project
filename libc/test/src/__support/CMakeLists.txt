add_libc_testsuite(libc_support_unittests)

add_libc_unittest(
  endian_test
  SUITE
    libc_support_unittests
  SRCS
    endian_test.cpp
  DEPENDS
    libc.src.__support.common
)

add_libc_unittest(
  high_precision_decimal_test
  SUITE
    libc_support_unittests
  SRCS
  high_precision_decimal_test.cpp
  DEPENDS
    libc.src.__support.high_precision_decimal
)
