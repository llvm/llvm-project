add_entrypoint_object(
  __errno_location
  SRCS
    errno_location.cpp
  HDRS
    llvmlibc_errno.h
)

add_custom_target(libc_errno_unittests)

add_libc_unittest(
  errno_test
  SUITE
    libc_errno_unittests
  SRCS
    errno_test.cpp
  DEPENDS
    __errno_location
)
