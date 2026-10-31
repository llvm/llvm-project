add_libc_testsuite(libc-sys-sysmacros-tests)

add_libc_test(
  sysmacros_test
  SUITE
    libc-sys-sysmacros-tests
  SRCS
    sysmacros_test.cpp
  DEPENDS
    libc.hdr.types.dev_t
    libc.src.sys.sysmacros.makedev
    libc.src.sys.sysmacros.major
    libc.src.sys.sysmacros.minor
)
