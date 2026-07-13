add_custom_target(libc_poll_unittests)

add_libc_unittest(
  poll_test
  SUITE
    libc_poll_unittests
  SRCS
    poll_test.cpp
  DEPENDS
    libc.hdr.limits_macros
    libc.src.errno.errno
    libc.src.poll.poll
    libc.test.UnitTest.ErrnoCheckingTest
)
