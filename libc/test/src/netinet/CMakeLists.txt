add_custom_target(libc_netinet_unittests)

add_libc_unittest(
  udp_test
  SUITE
    libc_netinet_unittests
  SRCS
    udp_test.cpp
  DEPENDS
    libc.hdr.types.struct_udphdr
    libc.src.arpa.inet.htons
    libc.src.string.memcmp
)
