add_custom_target(libc-fnmatch-tests)

add_libc_test(
  fnmatch_test
  SUITE
    libc-fnmatch-tests
  SRCS
    fnmatch_test.cpp
  DEPENDS
    libc.hdr.fnmatch_macros
    libc.src.fnmatch.fnmatch
)
