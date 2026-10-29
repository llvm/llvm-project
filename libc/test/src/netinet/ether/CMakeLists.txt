add_libc_testsuite(libc-netinet-ether-tests PARENT libc-netinet-tests)

add_libc_test(
  ether_ntoa_test
  SUITE
    libc-netinet-ether-tests
  SRCS
    ether_ntoa_test.cpp
  DEPENDS
    libc.hdr.types.struct_ether_addr
    libc.src.netinet.ether.ether_ntoa
)

add_libc_test(
  ether_ntoa_r_test
  SUITE
    libc-netinet-ether-tests
  SRCS
    ether_ntoa_r_test.cpp
  DEPENDS
    libc.hdr.types.struct_ether_addr
    libc.src.netinet.ether.ether_ntoa_r
)
