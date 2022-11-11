add_libc_testsuite(libc_sys_random_unittests)

add_libc_unittest(
  getrandom_test
  SUITE
    libc_sys_random_unittests
  SRCS
    getrandom_test.cpp
  DEPENDS
    libc.include.errno
    libc.include.math
    libc.include.sys_random
    libc.src.errno.errno
    libc.src.math.fabs
    libc.src.sys.random.getrandom
    libc.test.errno_setter_matcher
)
