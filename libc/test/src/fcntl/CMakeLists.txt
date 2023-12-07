add_custom_target(libc_fcntl_unittests)

add_subdirectory(testdata)

add_libc_unittest(
  creat_test
  SUITE
    libc_fcntl_unittests
  SRCS
    creat_test.cpp
  DEPENDS
    libc.include.fcntl
    libc.src.errno.errno
    libc.src.fcntl.creat
    libc.src.fcntl.open
    libc.src.unistd.close
    libc.test.errno_setter_matcher
)

add_libc_unittest(
  openat_test
  SUITE
    libc_fcntl_unittests
  SRCS
    openat_test.cpp
  DEPENDS
    libc.include.fcntl
    libc.src.errno.errno
    libc.src.fcntl.open
    libc.src.fcntl.openat
    libc.src.unistd.close
    libc.src.unistd.read
    libc.test.errno_setter_matcher
)
