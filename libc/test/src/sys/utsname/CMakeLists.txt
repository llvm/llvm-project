add_libc_testsuite(libc-sys-utsname-tests)

add_libc_test(
  uname_test
  SUITE
    libc-sys-utsname-tests
  SRCS
    uname_test.cpp
  DEPENDS
    libc.include.sys_utsname
    libc.src.__support.common
    libc.src.sys.utsname.uname
    libc.test.UnitTest.ErrnoSetterMatcher
)
