add_libc_integration_test_suite(libc-ucontext-integration-tests)

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
