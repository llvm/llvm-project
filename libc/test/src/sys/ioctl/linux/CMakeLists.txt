add_custom_target(libc_sys_ioctl_unittests)

add_libc_unittest(
  ioctl_test
  SUITE
    libc_sys_ioctl_unittests
  SRCS
    ioctl_test.cpp
  DEPENDS
    libc.hdr.sys_ioctl_macros
    libc.src.sys.ioctl.ioctl
    libc.src.errno.errno
    libc.src.fcntl.open
    libc.src.unistd.close
    libc.src.unistd.read
    libc.src.unistd.write
)

