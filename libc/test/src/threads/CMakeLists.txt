add_libc_testsuite(libc_threads_unittests)

add_libc_unittest(
  thrd_test
  SUITE
    libc_threads_unittests
  SRCS
    thrd_test.cpp
  DEPENDS
    __errno_location
    mmap
    munmap
    threads_h
    thrd_create
    thrd_join
)
