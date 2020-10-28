add_libc_testsuite(libc_sys_mman_unittests)

add_libc_unittest(
  mmap_test
  SUITE
    libc_sys_mman_unittests
  SRCS
    mmap_test.cpp
  DEPENDS
    libc.include.errno
    libc.include.sys_mman
    libc.src.errno.__errno_location
    libc.src.sys.mman.mmap
    libc.src.sys.mman.munmap
    libc.test.errno_setter_matcher
)
