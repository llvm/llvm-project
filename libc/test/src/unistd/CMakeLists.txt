add_libc_testsuite(libc_unistd_unittests)

add_libc_unittest(
  write_test
  SUITE
    libc_unistd_unittests
  SRCS
    write_test.cpp
  DEPENDS
    libc.src.unistd.write
    libc.include.errno
    libc.include.unistd
)
