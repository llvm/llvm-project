add_libc_testsuite(libc-support-net-tests PARENT libc-support-tests)

add_libc_test(
  hex_test
  SUITE
    libc-support-net-tests
  SRCS
    hex_test.cpp
  DEPENDS
    libc.hdr.stdint_proxy
    libc.src.__support.CPP.optional
    libc.src.__support.CPP.string_view
    libc.src.__support.macros.config
    libc.src.__support.net.hex
)

