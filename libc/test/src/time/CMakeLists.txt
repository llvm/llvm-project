add_libc_testsuite(libc_time_unittests)

add_libc_unittest(
  mktime
  SUITE
    libc_time_unittests
  SRCS
    gmtime_test.cpp
    mktime_test.cpp
  HDRS
    TmMatcher.h
  DEPENDS
    libc.src.time.gmtime
    libc.src.time.mktime
)
