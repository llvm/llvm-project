add_libc_testsuite(libc_time_unittests)

add_libc_unittest(
  gmtime
  SUITE
    libc_time_unittests
  SRCS
    gmtime_test.cpp
  HDRS
    TmMatcher.h
  DEPENDS
    libc.src.time.gmtime
)

add_libc_unittest(
  gmtime_r
  SUITE
    libc_time_unittests
  SRCS
    gmtime_r_test.cpp
  HDRS
    TmMatcher.h
  DEPENDS
    libc.src.time.gmtime_r
)

add_libc_unittest(
  mktime
  SUITE
    libc_time_unittests
  SRCS
    mktime_test.cpp
  HDRS
    TmMatcher.h
  DEPENDS
    libc.src.time.mktime
)
