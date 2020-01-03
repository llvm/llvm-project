add_custom_target(libc_errno_unittests)
add_dependencies(check_libc libc_errno_unittests)

add_libc_unittest(
  errno_test
  SUITE
    libc_errno_unittests
  SRCS
    errno_test.cpp
  DEPENDS
    __errno_location
)
