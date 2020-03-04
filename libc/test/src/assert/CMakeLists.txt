add_libc_testsuite(libc_assert_unittests)

add_libc_unittest(
  assert_test
  SUITE
    libc_assert_unittests
  SRCS
    assert_test.cpp
  DEPENDS
    libc.src.assert.__assert_fail
    # These are necessary for now because dependencies are not properly added.
    libc.src.signal.raise
    libc.src.stdlib._Exit
    libc.src.stdlib.abort
)
