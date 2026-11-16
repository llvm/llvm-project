add_libc_testsuite(libc-netdb-tests)

add_libc_test(
  gai_strerror_test
  SUITE
    libc-netdb-tests
  SRCS
    gai_strerror_test.cpp
  DEPENDS
    libc.hdr.netdb_macros
    libc.src.netdb.gai_strerror
)

add_libc_test(
  getaddrinfo_test
  SUITE
    libc-netdb-tests
  SRCS
    getaddrinfo_test.cpp
  DEPENDS
    libc.hdr.errno_macros
    libc.hdr.netdb_macros
    libc.hdr.sys_socket_macros
    libc.hdr.types.struct_addrinfo
    libc.src.netdb.freeaddrinfo
    libc.src.netdb.getaddrinfo
    libc.test.UnitTest.ErrnoCheckingTest
    libc.test.UnitTest.ErrnoSetterMatcher
)

if(NOT TARGET libc.src.netdb.getprotobyname_r)
  return()
endif()

add_libc_test(
  getprotobyname_r_test
  SUITE
    libc-netdb-tests
  SRCS
    getprotobyname_r_test.cpp
  DEPENDS
    libc.hdr.errno_macros
    libc.hdr.signal_macros
    libc.hdr.types.size_t
    libc.hdr.types.struct_protoent
    libc.src.__support.CPP.string_view
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
    libc.src.netdb.getprotobyname_r
    libc.src.stdio.remove
    libc.src.string.string_utils
    libc.test.UnitTest.ErrnoCheckingTest
)
