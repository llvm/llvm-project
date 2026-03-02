add_custom_target(libc_arpa_inet_unittests)

add_libc_unittest(
  htonl
  SUITE
    libc_arpa_inet_unittests
  SRCS
    htonl_test.cpp
  DEPENDS
    libc.src.arpa.inet.htonl
    libc.src.arpa.inet.ntohl
)

add_libc_unittest(
  htons
  SUITE
    libc_arpa_inet_unittests
  SRCS
    htons_test.cpp
  DEPENDS
    libc.src.arpa.inet.htons
    libc.src.arpa.inet.ntohs
)

add_libc_unittest(
  inet_addr
  SUITE
    libc_arpa_inet_unittests
  SRCS
    inet_addr_test.cpp
  DEPENDS
    libc.src.arpa.inet.htonl
    libc.src.arpa.inet.inet_addr
)

add_libc_unittest(
  inet_aton
  SUITE
    libc_arpa_inet_unittests
  SRCS
    inet_aton_test.cpp
  DEPENDS
    libc.src.arpa.inet.htonl
    libc.src.arpa.inet.inet_aton
)

add_libc_unittest(
  ntohl
  SUITE
    libc_arpa_inet_unittests
  SRCS
    ntohl_test.cpp
  DEPENDS
    libc.src.arpa.inet.htonl
    libc.src.arpa.inet.ntohl
)

add_libc_unittest(
  ntohs
  SUITE
    libc_arpa_inet_unittests
  SRCS
    ntohs_test.cpp
  DEPENDS
    libc.src.arpa.inet.htons
    libc.src.arpa.inet.ntohs
)
