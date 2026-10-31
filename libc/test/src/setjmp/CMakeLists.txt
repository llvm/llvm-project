add_libc_testsuite(libc-setjmp-tests)

# Skip setjmp tests under sanitizers
if(LLVM_USE_SANITIZER)
  return()
endif()

add_libc_test(
  setjmp_test
  SUITE
    libc-setjmp-tests
  SRCS
    setjmp_test.cpp
  CXX_STANDARD
    20
  DEPENDS
    libc.src.setjmp.longjmp
    libc.src.setjmp.setjmp
)

add_libc_test(
  sigsetjmp_test
  SUITE
    libc-setjmp-tests
  SRCS
    sigsetjmp_test.cpp
  CXX_STANDARD
    20
  DEPENDS
    libc.src.setjmp.sigsetjmp
    libc.src.setjmp.siglongjmp
    libc.src.signal.sigprocmask
    libc.src.string.memset
    libc.src.string.memcmp
    libc.hdr.types.sigset_t
)
