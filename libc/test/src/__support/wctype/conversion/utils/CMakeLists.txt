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

add_libc_test(
  shared_utils_test
  SUITE
    libc-support-wctype-conversion-utils-tests
  SRCS
    shared_utils_test.cpp
  DEPENDS
    libc.src.__support.integer_literals
    libc.src.__support.wctype.conversion.utils.utils
)
