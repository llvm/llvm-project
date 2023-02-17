add_libc_testsuite(libc_network_unittests)

add_libc_unittest(
  htonl
  SUITE
    libc_network_unittests
  SRCS
    htonl_test.cpp
  CXX_STANDARD
    20
  DEPENDS
    libc.src.network.htonl
)

add_libc_unittest(
  htons
  SUITE
    libc_network_unittests
  SRCS
    htons_test.cpp
  CXX_STANDARD
    20
  DEPENDS
    libc.src.network.htons
)
