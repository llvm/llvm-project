add_custom_target(libc_sys_mman_unittests)
add_dependencies(check_libc libc_sys_mman_unittests)

add_libc_unittest(
  mmap_test
  SUITE
    libc_sys_mman_unittests
  SRCS
    mmap_test.cpp
  DEPENDS
    mmap
    munmap
    __errno_location
)
