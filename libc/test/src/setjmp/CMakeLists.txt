add_libc_testsuite(libc_setjmp_unittests)

# Skip setjmp tests under sanitizers
if(LLVM_USE_SANITIZER)
  return
endif()

add_libc_unittest(
  setjmp_test
  SUITE
    libc_setjmp_unittests
  SRCS
    setjmp_test.cpp
  DEPENDS
    libc.include.setjmp
    libc.src.setjmp.longjmp
    libc.src.setjmp.setjmp
)
