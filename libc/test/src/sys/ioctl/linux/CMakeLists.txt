add_custom_target(libc_sys_ioctl_unittests)

add_libc_unittest(
  ioctl_test
  SUITE
    libc_sys_ioctl_unittests
  SRCS
    ioctl_test.cpp
  DEPENDS
    libc.include.sys_ioctl
    libc.src.errno.errno
    libc.test.errno_setter_matcher
)

