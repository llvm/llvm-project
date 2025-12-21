add_custom_target(libc_sys_prctl_unittests)

# Temporarily disable this test while setting up arm and riscv buildbots
# using qemu, since PR_GET_THP_DISABLE is not supported on qemu.
if (NOT (LIBC_TARGET_ARCHITECTURE_IS_ARM OR
	 LIBC_TARGET_ARCHITECTURE_IS_RISCV32 OR
	 LIBC_TARGET_ARCHITECTURE_IS_RISCV64))
add_libc_unittest(
  prctl_test
  SUITE
    libc_sys_prctl_unittests
  SRCS
    prctl_test.cpp
  DEPENDS
    libc.include.sys_prctl
    libc.src.sys.prctl.prctl
    libc.src.errno.errno
    libc.test.UnitTest.ErrnoCheckingTest
    libc.test.UnitTest.ErrnoSetterMatcher
)
endif()
