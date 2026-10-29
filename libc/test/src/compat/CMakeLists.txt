add_libc_testsuite(libc-compat-tests)

add_libc_test(
  syscall_test
  SUITE
    libc-compat-tests
  SRCS
    syscall_test.cpp
  DEPENDS
    libc.src.__support.OSUtil.osutil
    libc.src.__support.macros.properties.architectures
    libc.src.compat.syscall
)
