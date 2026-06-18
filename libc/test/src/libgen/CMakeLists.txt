add_custom_target(libc-libgen-tests)

add_libc_test(
  basename_test
  SUITE
    libc-libgen-tests
  SRCS
    basename_test.cpp
  DEPENDS
    libc.src.libgen.basename
)

add_libc_test(
  dirname_test
  SUITE
    libc-libgen-tests
  SRCS
    dirname_test.cpp
  DEPENDS
    libc.src.libgen.dirname
)
