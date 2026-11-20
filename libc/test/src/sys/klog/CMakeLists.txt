add_libc_testsuite(libc-sys-klog-tests)

add_libc_test(
  klogctl_test
  SUITE
    libc-sys-klog-tests
  SRCS
    klogctl_test.cpp
  DEPENDS
    libc.hdr.errno_macros
    libc.src.__support.libc_errno
    libc.src.sys.klog.klogctl
    libc.test.UnitTest.ErrnoCheckingTest
)
