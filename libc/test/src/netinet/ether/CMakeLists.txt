add_custom_target(libc_netinet_ether_unittests)

add_libc_test(
  ether_ntoa_test
  SUITE
    libc_netinet_ether_unittests
  SRCS
    ether_ntoa_test.cpp
  DEPENDS
    libc.hdr.types.struct_ether_addr
    libc.src.netinet.ether.ether_ntoa
)

add_libc_test(
  ether_ntoa_r_test
  SUITE
    libc_netinet_ether_unittests
  SRCS
    ether_ntoa_r_test.cpp
  DEPENDS
    libc.hdr.types.struct_ether_addr
    libc.src.netinet.ether.ether_ntoa_r
)
