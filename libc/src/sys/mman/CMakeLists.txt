#TODO: The sources and target listed here should ideally live in config/linux.

add_entrypoint_object(
  mmap
  SRCS
    mmap.cpp
  HDRS
    mmap.h
  DEPENDS
    sys_mman_h
    sys_syscall_h
    syscall_impl_h
    __errno_location
)

add_entrypoint_object(
  munmap
  SRCS
    munmap.cpp
  HDRS
    munmap.h
  DEPENDS
    sys_mman_h
    sys_syscall_h
    syscall_impl_h
    __errno_location
)

add_custom_target(libc_sys_mman_unittests)

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
