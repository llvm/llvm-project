add_custom_target(libc-glob-tests)

add_libc_test(
  glob_test
  SUITE
    libc-glob-tests
  SRCS
    glob_test.cpp
  DEPENDS
    libc.hdr.glob_macros
    libc.hdr.types.glob_t
    libc.src.glob.glob
    libc.src.glob.globfree
)
