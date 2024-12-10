add_custom_target(libc_sys_random_unittests)

add_libc_unittest(
  getrandom_test
  SUITE
    libc_sys_random_unittests
  SRCS
    getrandom_test.cpp
  DEPENDS
    libc.hdr.sys_random_macros
    libc.src.errno.errno
    libc.src.math.fabs
    libc.src.sys.random.getrandom
    libc.test.UnitTest.ErrnoSetterMatcher
)
