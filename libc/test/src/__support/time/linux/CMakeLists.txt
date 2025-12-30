add_libc_test(
  timeout_test
  SUITE libc-support-time-tests
  SRCS timeout_test.cpp
  DEPENDS
    libc.src.__support.time.abs_timeout
    libc.src.__support.time.monotonicity
    libc.src.__support.CPP.expected
)
