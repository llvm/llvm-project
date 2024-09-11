add_custom_target(libc_ftw_unittests)

add_libc_unittest(
  ftw_test
  SUITE
    libc_ftw_unittests
  SRCS
    ftw_test.cpp
  DEPENDS
    libc.src.errno.errno
)
