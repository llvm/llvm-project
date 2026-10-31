add_libc_testsuite(libc-err-tests)

if(NOT TARGET libc.src.__support.OSUtil.osutil)
  return()
endif()

add_libc_test(
  err_test
  SUITE
    libc-err-tests
  SRCS
    err_test.cpp
  DEPENDS
    libc.src.err.err
    libc.src.__support.libc_errno
)

add_libc_test(
  errx_test
  SUITE
    libc-err-tests
  SRCS
    errx_test.cpp
  DEPENDS
    libc.src.err.errx
)

add_libc_test(
  verr_test
  SUITE
    libc-err-tests
  SRCS
    verr_test.cpp
  DEPENDS
    libc.src.err.verr
    libc.src.__support.libc_errno
)

add_libc_test(
  verrx_test
  SUITE
    libc-err-tests
  SRCS
    verrx_test.cpp
  DEPENDS
    libc.src.err.verrx
)

add_libc_test(
  warn_test
  SUITE
    libc-err-tests
  SRCS
    warn_test.cpp
  DEPENDS
    libc.src.err.warn
    libc.src.__support.libc_errno
)

add_libc_test(
  warnx_test
  SUITE
    libc-err-tests
  SRCS
    warnx_test.cpp
  DEPENDS
    libc.src.err.warnx
    libc.src.__support.libc_errno
)

add_libc_test(
  vwarn_test
  SUITE
    libc-err-tests
  SRCS
    vwarn_test.cpp
  DEPENDS
    libc.src.err.vwarn
    libc.src.__support.libc_errno
)

add_libc_test(
  vwarnx_test
  SUITE
    libc-err-tests
  SRCS
    vwarnx_test.cpp
  DEPENDS
    libc.src.err.vwarnx
)
