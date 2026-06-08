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
    libc.src.__support.libc_errno
)

add_libc_test(
  errx_test
  UNIT_TEST_ONLY
  SUITE
    libc_err_unittests
  SRCS
    errx_test.cpp
  DEPENDS
    libc.src.err.errx
)

add_libc_test(
  verr_test
  UNIT_TEST_ONLY
  SUITE
    libc_err_unittests
  SRCS
    verr_test.cpp
  DEPENDS
    libc.src.err.verr
    libc.src.__support.libc_errno
)

add_libc_test(
  verrx_test
  UNIT_TEST_ONLY
  SUITE
    libc_err_unittests
  SRCS
    verrx_test.cpp
  DEPENDS
    libc.src.err.verrx
)

add_libc_test(
  warn_test
  UNIT_TEST_ONLY
  SUITE
    libc_err_unittests
  SRCS
    warn_test.cpp
  DEPENDS
    libc.src.err.warn
    libc.src.__support.libc_errno
)

add_libc_test(
  warnx_test
  UNIT_TEST_ONLY
  SUITE
    libc_err_unittests
  SRCS
    warnx_test.cpp
  DEPENDS
    libc.src.err.warnx
    libc.src.__support.libc_errno
)

add_libc_test(
  vwarn_test
  UNIT_TEST_ONLY
  SUITE
    libc_err_unittests
  SRCS
    vwarn_test.cpp
  DEPENDS
    libc.src.err.vwarn
    libc.src.__support.libc_errno
)

add_libc_test(
  vwarnx_test
  UNIT_TEST_ONLY
  SUITE
    libc_err_unittests
  SRCS
    vwarnx_test.cpp
  DEPENDS
    libc.src.err.vwarnx
)
