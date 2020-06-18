add_libc_testsuite(libc_stdlib_unittests)

add_libc_unittest(
  _Exit_test
  SUITE
    libc_stdlib_unittests
  SRCS
    _Exit_test.cpp
  DEPENDS
    libc.include.stdlib
    libc.src.stdlib._Exit
)

add_libc_unittest(
  abort_test
  SUITE
    libc_stdlib_unittests
  SRCS
    abort_test.cpp
  DEPENDS
    libc.include.stdlib
    libc.include.signal
    libc.src.stdlib.abort
    libc.src.stdlib._Exit
    libc.src.signal.raise
)
