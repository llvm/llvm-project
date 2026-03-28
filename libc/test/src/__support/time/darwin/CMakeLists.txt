add_libc_test(
  clock_gettime
  SUITE libc-support-time-tests
  SRCS clock_gettime.cpp
  DEPENDS
    libc.src.__support.CPP.expected
    libc.src.__support.time.darwin.clock_gettime
)
