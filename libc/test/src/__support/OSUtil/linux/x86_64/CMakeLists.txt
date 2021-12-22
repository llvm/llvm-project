add_libc_unittest(
  syscall_unittest
  SUITE libc_osutil_tests
  SRCS syscall_test.cpp
  DEPENDS
   libc.src.__support.OSUtil.osutil
)
