add_custom_target(libc-support-wctype-tests)

set(CONSTEXPR_FLAGS "")

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
    list(APPEND CONSTEXPR_FLAGS
      -Xclang
      -fconstexpr-steps=10000000)
  else()
    list(APPEND CONSTEXPR_FLAGS
      -fconstexpr-steps=10000000)
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  list(APPEND CONSTEXPR_FLAGS
    -fconstexpr-ops-limit=10000000)
endif()

add_libc_test(
  wctype_perfect_hash_test
  SUITE
    libc-support-wctype-tests
  SRCS
    wctype_perfect_hash_test.cpp
  COMPILE_OPTIONS
    ${CONSTEXPR_FLAGS}
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
