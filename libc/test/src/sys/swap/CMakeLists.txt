add_libc_testsuite(libc-sys-swap-tests)

add_libc_test(
  swapon_swapoff_test
  SUITE
    libc-sys-swap-tests
  SRCS
    swapon_swapoff_test.cpp
  DEPENDS
    libc.hdr.errno_macros
    libc.src.sys.swap.swapoff
    libc.src.sys.swap.swapon
    libc.test.UnitTest.ErrnoCheckingTest
    libc.test.UnitTest.ErrnoSetterMatcher
)
