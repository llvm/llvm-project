add_custom_target(libc-support-wctype-tests)

add_libc_test(
  wctype_classification_utils_test
  SUITE
    libc-support-tests
  SRCS
    wctype_classification_utils_test.cpp
  DEPENDS
    libc.hdr.stdint_proxy
    libc.src.__support.wctype.wctype_classification_utils
)

