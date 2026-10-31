add_libc_testsuite(libc-ucontext-tests)

if(TARGET libc.src.ucontext.getcontext)
  add_libc_test(
    ucontext_test
    SUITE
      libc-ucontext-tests
    SRCS
      ucontext_test.cpp
    DEPENDS
      libc.src.ucontext.getcontext
      libc.src.ucontext.setcontext
      libc.src.signal.sigemptyset
      libc.src.signal.sigaddset
      libc.src.signal.sigismember
      libc.src.signal.sigprocmask
  )
endif()
