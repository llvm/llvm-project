add_libc_testsuite(libc-sys-sysinfo-tests)

add_libc_test(
  sysinfo_test
  SUITE
    libc-sys-sysinfo-tests
  SRCS
    sysinfo_test.cpp
  DEPENDS
    libc.hdr.errno_macros
    libc.hdr.types.struct_sysinfo
    libc.include.sys_sysinfo
    libc.src.__support.common
    libc.src.sys.sysinfo.sysinfo
    libc.test.UnitTest.ErrnoSetterMatcher
)
