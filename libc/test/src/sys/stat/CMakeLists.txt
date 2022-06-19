add_libc_testsuite(libc_sys_stat_unittests)

add_subdirectory(testdata)

add_libc_unittest(
  mkdirat_test
  SUITE
    libc_sys_stat_unittests
  SRCS
    mkdirat_test.cpp
  DEPENDS
    libc.include.errno
    libc.include.fcntl
    libc.include.sys_stat
    libc.src.sys.stat.mkdirat
    libc.src.unistd.rmdir
)
