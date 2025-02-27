add_custom_target(libc-strings-tests)

add_libc_test(
  bcopy_test
  SUITE
    libc-strings-tests
  SRCS
    bcopy_test.cpp
  DEPENDS
    libc.src.strings.bcopy
  LINK_LIBRARIES
    LibcMemoryHelpers
)

add_libc_test(
  index_test
  SUITE
    libc-strings-tests
  SRCS
    index_test.cpp
  DEPENDS
    libc.src.strings.index
    libc.test.src.string.strchr_test_support
)

add_libc_test(
  rindex_test
  SUITE
    libc-strings-tests
  SRCS
    rindex_test.cpp
  DEPENDS
    libc.src.strings.rindex
    libc.test.src.string.strchr_test_support
)

add_libc_test(
  strcasecmp_test
  SUITE
    libc-strings-tests
  SRCS
    strcasecmp_test.cpp
  DEPENDS
    libc.src.strings.strcasecmp
)

add_libc_test(
  strncasecmp_test
  SUITE
    libc-strings-tests
  SRCS
    strncasecmp_test.cpp
  DEPENDS
    libc.src.strings.strncasecmp
)

add_libc_multi_impl_test(bcmp libc-strings-tests SRCS bcmp_test.cpp)
add_libc_multi_impl_test(bzero libc-strings-tests SRCS bzero_test.cpp)
