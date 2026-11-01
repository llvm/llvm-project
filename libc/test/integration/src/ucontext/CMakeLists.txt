add_libc_testsuite(libc-ucontext-integration-tests PARENT libc-integration-tests)

add_integration_test(
  ucontext_test
  SUITE
    libc-ucontext-integration-tests
  SRCS
    ucontext_test.cpp
  DEPENDS
    libc.src.ucontext.getcontext
    libc.src.ucontext.setcontext
)
