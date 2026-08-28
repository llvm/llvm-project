add_custom_target(libc-netdb-tests)

add_libc_test(
  netdb_test
  SUITE
    libc-netdb-tests
  SRCS
    netdb_test.cpp
  DEPENDS
    libc.hdr.errno_macros
    libc.hdr.netdb_macros
    libc.hdr.types.struct_addrinfo
    libc.src.errno.errno
    libc.src.netdb.freeaddrinfo
    libc.src.netdb.getaddrinfo
    libc.test.UnitTest.ErrnoCheckingTest
    libc.test.UnitTest.ErrnoSetterMatcher
)
