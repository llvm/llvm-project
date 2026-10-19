add_custom_target(libc_sys_sysmacros_unittests)

add_libc_test(
  sysmacros_test
  SUITE
    libc_sys_sysmacros_unittests
  SRCS
    sysmacros_test.cpp
  DEPENDS
    libc.hdr.types.dev_t
    libc.src.sys.sysmacros.makedev
    libc.src.sys.sysmacros.major
    libc.src.sys.sysmacros.minor
)
