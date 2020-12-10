add_libc_testsuite(libc_time_unittests)

add_libc_unittest(
  mktime
  SUITE
    libc_time_unittests
  SRCS
    mktime_test.cpp
  DEPENDS
    libc.src.time.mktime
)
