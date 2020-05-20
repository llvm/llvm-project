add_libc_testsuite(libc_stdio_unittests)

add_libc_unittest(
  fwrite_test
  SUITE
    libc_stdio_unittests
  SRCS
    fwrite_test.cpp
  DEPENDS
    libc.src.stdio.fwrite
)
