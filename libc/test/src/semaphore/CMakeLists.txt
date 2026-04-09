add_custom_target(libc_semaphore_unittests)

add_libc_unittest(
  semaphore_test
  SUITE
    libc_semaphore_unittests
  SRCS
    semaphore_test.cpp
  DEPENDS
    libc.src.semaphore.posix_semaphore
)
