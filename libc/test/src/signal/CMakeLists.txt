add_libc_testsuite(libc_signal_unittests)

add_libc_unittest(
  raise_test
  SUITE
    libc_signal_unittests
  SRCS
    raise_test.cpp
  DEPENDS
    raise
    signal_h
)
