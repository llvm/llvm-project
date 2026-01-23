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
  ffs_test
  SUITE
    libc-strings-tests
  SRCS
    ffs_test.cpp
  DEPENDS
    libc.src.strings.ffs
)

add_libc_test(
  ffsl_test
  SUITE
    libc-strings-tests
  SRCS
    ffsl_test.cpp
  DEPENDS
    libc.src.strings.ffsl
)

add_libc_test(
  ffsll_test
  SUITE
    libc-strings-tests
  SRCS
    ffsll_test.cpp
  DEPENDS
    libc.src.strings.ffsll
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
  strcasecmp_l_test
  SUITE
    libc-strings-tests
  SRCS
    strcasecmp_l_test.cpp
  DEPENDS
    libc.hdr.locale_macros
    libc.src.locale.freelocale
    libc.src.locale.newlocale
    libc.src.strings.strcasecmp_l
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

add_libc_test(
  strncasecmp_l_test
  SUITE
    libc-strings-tests
  SRCS
    strncasecmp_l_test.cpp
  DEPENDS
    libc.hdr.locale_macros
    libc.src.locale.freelocale
    libc.src.locale.newlocale
    libc.src.strings.strncasecmp_l
)

add_libc_test(
  wide_read_memory_test
  SUITE
    libc-strings-tests
  SRCS
    wide_read_memory_test.cpp
  DEPENDS
    libc.src.string.string_utils
    libc.src.sys.mman.mmap
    libc.src.sys.mman.mprotect
    libc.src.sys.mman.munmap
    libc.src.unistd.linux.getpagesize
    libc.src.__support.CPP.array
)

add_libc_multi_impl_test(bcmp libc-strings-tests SRCS bcmp_test.cpp)
add_libc_multi_impl_test(bzero libc-strings-tests SRCS bzero_test.cpp)
