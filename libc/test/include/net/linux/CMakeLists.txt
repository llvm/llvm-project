add_libc_test(
  net_if_and_linux_if_test
  SUITE
    libc_include_tests
  SRCS
    linux_if_helper.cpp
    net_if_and_linux_if_test.cpp
  DEPENDS
    libc.include.net_if
    libc.include.sys_socket
)

