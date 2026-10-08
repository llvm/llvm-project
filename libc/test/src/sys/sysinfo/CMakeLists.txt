add_custom_target(libc_sys_sysinfo_unittests)

add_libc_test(
  sysinfo_test
  SUITE
    libc_sys_sysinfo_unittests
  SRCS
    sysinfo_test.cpp
  DEPENDS
    libc.hdr.errno_macros
    libc.hdr.types.struct_sysinfo
    libc.include.sys_sysinfo
    libc.src.errno.errno
    libc.src.__support.common
    libc.src.sys.sysinfo.sysinfo
    libc.test.UnitTest.ErrnoSetterMatcher
)
