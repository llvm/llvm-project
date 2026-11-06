add_libc_testsuite(libc-sys-shm-tests)

add_libc_test(
  shmget_test
  SUITE
    libc-sys-shm-tests
  SRCS
    shmget_test.cpp
  DEPENDS
    libc.hdr.errno_macros
    libc.hdr.sys_ipc_macros
    libc.hdr.sys_stat_macros
    libc.src.sys.shm.shmget
    libc.test.UnitTest.ErrnoCheckingTest
    libc.test.UnitTest.ErrnoSetterMatcher
)
