add_custom_target(libc-support-wctype-conversion-utils-tests)

add_libc_test(
  slice_test
  SUITE
    libc-support-wctype-conversion-utils-tests
  SRCS
    slice_test.cpp
  DEPENDS
    libc.src.__support.wctype.conversion.utils.slice
)
