add_libc_testsuite(libc-elf-tests)

add_libc_test(
  dynamic_test
  SUITE
    libc-support-tests
  SRCS
    dynamic_test.cpp
  DEPENDS
    libc.src.__support.elf.dynamic
)
