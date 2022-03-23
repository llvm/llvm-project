add_libc_testsuite(libc_stdio_unittests)

add_libc_unittest(
  fileop_test
  SUITE
    libc_stdio_unittests
  SRCS
    fileop_test.cpp
  DEPENDS
    libc.src.stdio.fclose
    libc.src.stdio.fopen
    libc.src.stdio.fread
    libc.src.stdio.fseek
    libc.src.stdio.fwrite
)

add_subdirectory(testdata)
