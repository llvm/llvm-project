add_libc_testsuite(libc_stdlib_unittests)

add_libc_unittest(
  _Exit_test
  SUITE
    libc_stdlib_unittests
  SRCS
    _Exit_test.cpp
  DEPENDS
    stdlib_h
    _Exit
)

add_libc_unittest(
  abort_test
  SUITE
    libc_stdlib_unittests
  SRCS
    abort_test.cpp
  DEPENDS
    stdlib_h
    signal_h
    abort
    _Exit
    raise
)
