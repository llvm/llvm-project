add_libc_testsuite(libc-assert-tests)

add_libc_test(
  assert_test
  SUITE
    libc-assert-tests
  SRCS
    assert_test.cpp
  DEPENDS
    libc.src.assert.__assert_fail
    libc.src.unistd.close
    # These are necessary for now because dependencies are not properly added.
    libc.src.signal.raise
    libc.src.stdlib._Exit
    libc.src.stdlib.abort
)
