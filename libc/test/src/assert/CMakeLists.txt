add_libc_testsuite(libc_assert_unittests)

add_libc_unittest(
  assert_test
  SUITE
    libc_assert_unittests
  SRCS
    assert_test.cpp
  DEPENDS
    __assert_fail
    # These are necessary for now because dependencies are not properly added.
    abort
    raise
    _Exit
)
