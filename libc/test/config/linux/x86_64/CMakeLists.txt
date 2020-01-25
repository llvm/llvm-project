add_libc_unittest(
  libc_linux_x86_64_syscall_unittest
  SUITE libc_linux_tests
  SRCS syscall_test.cpp
  DEPENDS
   linux_syscall_h
   support_common_h
)
