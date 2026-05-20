add_custom_target(libc-support-wctype-tests)

add_libc_test(
  wctype_perfect_hash_test
  SUITE
    libc-support-wctype-tests
  SRCS
    wctype_perfect_hash_test.cpp
  COMPILE_OPTIONS
    $<$<CXX_COMPILER_ID:Clang>:-Xclang -fconstexpr-steps=10000000>
    $<$<CXX_COMPILER_ID:GNU>:-fconstexpr-ops-limit=10000000>
  DEPENDS
    libc.src.__support.wctype.lower_to_upper
    libc.src.__support.wctype.upper_to_lower
)

add_libc_test(
  wctype_classification_utils_test
  SUITE
    libc-support-wctype-tests
  SRCS
    wctype_classification_utils_test.cpp
  DEPENDS
    libc.hdr.stdint_proxy
    libc.src.__support.wctype.wctype_classification_utils
)
