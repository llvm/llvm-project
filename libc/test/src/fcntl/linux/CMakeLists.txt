add_custom_target(libc_fallocate_unittests)

add_libc_test(
  fallocate_test
  SUITE
    libc_fcntl_unittests
  SRCS
    fallocate_test.cpp
  DEPENDS
    libc.include.fcntl
    libc.src.fcntl.fallocate
    libc.src.fcntl.open
    libc.src.unistd.close
    libc.src.unistd.unlink
    libc.src.sys.stat.fstat
    libc.src.__support.libc_errno
)
