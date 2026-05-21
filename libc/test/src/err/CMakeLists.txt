add_custom_target(libc_err_unittests)

if(NOT TARGET libc.src.__support.OSUtil.osutil)
  return()
endif()

add_libc_test(
  err_test
  UNIT_TEST_ONLY
  SUITE
    libc_err_unittests
  SRCS
    err_test.cpp
  DEPENDS
    libc.src.err.err
    libc.src.err.errx
    libc.src.err.verr
    libc.src.err.verrx
    libc.src.err.warn
    libc.src.err.warnx
    libc.src.err.vwarn
    libc.src.err.vwarnx
    libc.src.__support.libc_errno
    libc.src.__support.common
)
