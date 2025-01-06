add_custom_target(libc_sys_prctl_unittests)

add_libc_unittest(
  prctl_test
  SUITE
    libc_sys_prctl_unittests
  SRCS
    prctl_test.cpp
  DEPENDS
    libc.include.sys_prctl
    libc.src.sys.prctl.prctl
    libc.src.errno.errno
)
