add_libc_testsuite(libc-sys-sendfile-tests)

add_subdirectory(testdata)

add_libc_test(
  sendfile_test
  SUITE
    libc-sys-sendfile-tests
  SRCS
    sendfile_test.cpp
  DEPENDS
    libc.hdr.errno_macros
    libc.hdr.fcntl_macros
    libc.hdr.sys_stat_macros
    libc.src.fcntl.open
    libc.src.sys.sendfile.sendfile
    libc.src.unistd.close
    libc.src.unistd.read
    libc.src.unistd.unlink
    libc.src.unistd.write
    libc.test.UnitTest.ErrnoCheckingTest
    libc.test.UnitTest.ErrnoSetterMatcher
)
