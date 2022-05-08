add_libc_unittest(
  syscall_unittest
  SUITE libc_osutil_tests
  SRCS syscall_test.cpp
  COMPILE_OPTIONS
    -Wno-unused-variable # Only signature tests, declared variables are unused.
  DEPENDS
    libc.src.__support.OSUtil.osutil
)
