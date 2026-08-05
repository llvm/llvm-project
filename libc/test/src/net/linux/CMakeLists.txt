add_custom_target(libc_net_unittests)

add_libc_unittest(
  if_indextoname_test
  SUITE
    libc_net_unittests
  SRCS
    if_indextoname_test.cpp
  DEPENDS
    libc.hdr.errno_macros
    libc.hdr.net_if_macros
    libc.src.net.if_indextoname
    libc.src.net.if_nametoindex
    libc.test.UnitTest.ErrnoCheckingTest
)

add_libc_unittest(
  if_nametoindex_test
  SUITE
    libc_net_unittests
  SRCS
    if_nametoindex_test.cpp
  DEPENDS
    libc.hdr.errno_macros
    libc.src.net.if_nametoindex
    libc.test.UnitTest.ErrnoCheckingTest
)
