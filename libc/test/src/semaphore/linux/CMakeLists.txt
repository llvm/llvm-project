add_custom_target(libc_semaphore_unittests)

add_libc_unittest(
  semaphore_test
  SUITE
    libc_semaphore_unittests
  SRCS
    semaphore_test.cpp
  DEPENDS
    libc.hdr.errno_macros
    libc.hdr.fcntl_macros
    libc.src.semaphore.linux.semaphore
    libc.src.semaphore.linux.named_semaphore
)
