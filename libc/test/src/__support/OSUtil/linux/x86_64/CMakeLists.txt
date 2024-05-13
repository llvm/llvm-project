add_libc_test(
  syscall_test
  SUITE libc-osutil-tests
  SRCS syscall_test.cpp
  COMPILE_OPTIONS
    -Wno-unused-variable # Only signature tests, declared variables are unused.
  DEPENDS
    libc.src.__support.OSUtil.osutil
)
