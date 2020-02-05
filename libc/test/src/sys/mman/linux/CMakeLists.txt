add_libc_testsuite(libc_sys_mman_unittests)

add_libc_unittest(
  mmap_test
  SUITE
    libc_sys_mman_unittests
  SRCS
    mmap_test.cpp
  DEPENDS
    errno_h
    sys_mman_h
    mmap
    munmap
    __errno_location
)
