add_libc_testsuite(libc_unistd_unittests)

add_libc_unittest(
  read_write_test
  SUITE
    libc_unistd_unittests
  SRCS
    read_write_test.cpp
  DEPENDS
    libc.include.errno
    libc.include.unistd
    libc.src.fcntl.open
    libc.src.unistd.close
    libc.src.unistd.fsync
    libc.src.unistd.read
    libc.src.unistd.write
    libc.test.errno_setter_matcher
)
