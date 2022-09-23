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
    libc.src.errno.errno
    libc.src.sys.mman.mmap
    libc.src.sys.mman.munmap
    libc.test.errno_setter_matcher
)

add_libc_unittest(
  mprotect_test
  SUITE
    libc_sys_mman_unittests
  SRCS
    mprotect_test.cpp
  DEPENDS
    libc.include.errno
    libc.include.sys_mman
    libc.include.signal
    libc.src.errno.errno
    libc.src.sys.mman.mmap
    libc.src.sys.mman.munmap
    libc.src.sys.mman.mprotect
    libc.test.errno_setter_matcher
)

