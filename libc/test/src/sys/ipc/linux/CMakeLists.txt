add_libc_testsuite(libc-sys-ipc-tests)

add_libc_test(
  ftok_test
  SUITE
    libc-sys-ipc-tests
  SRCS
    ftok_test.cpp
  DEPENDS
    libc.hdr.fcntl_macros
    libc.include.sys_ipc
    libc.src.fcntl.open
    libc.src.sys.ipc.ftok
    libc.src.unistd.close
    libc.src.unistd.unlink
    libc.test.UnitTest.ErrnoCheckingTest
    libc.test.UnitTest.ErrnoSetterMatcher
)
