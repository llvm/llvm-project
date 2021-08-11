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

add_libc_unittest(
  abs_test
  SUITE
    libc_stdlib_unittests
  SRCS
    abs_test.cpp
  DEPENDS
    libc.src.stdlib.abs
)

add_libc_unittest(
  labs_test
  SUITE
    libc_stdlib_unittests
  SRCS
    labs_test.cpp
  DEPENDS
    libc.src.stdlib.labs
)

add_libc_unittest(
  llabs_test
  SUITE
    libc_stdlib_unittests
  SRCS
    llabs_test.cpp
  DEPENDS
    libc.src.stdlib.llabs
)

add_libc_unittest(
  strtoll_test
  SUITE
    libc_stdlib_unittests
  SRCS
    strtoll_test.cpp
  DEPENDS
    libc.src.stdlib.strtoll
    libc.include.errno
    libc.test.errno_setter_matcher
)
