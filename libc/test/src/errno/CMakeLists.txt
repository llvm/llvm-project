if(NOT LLVM_LIBC_FULL_BUILD OR LIBC_TARGET_OS_IS_GPU)
  return()
endif()

add_libc_testsuite(libc-errno-tests)

add_libc_test(
  errno_test
  SUITE
    libc-errno-tests
  SRCS
    errno_test.cpp
  DEPENDS
    libc.src.errno.errno
    libc.test.UnitTest.ErrnoCheckingTest
)
