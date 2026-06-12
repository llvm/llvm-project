add_custom_target(libc_ucontext_unittests)

if(TARGET libc.src.ucontext.getcontext)
  add_libc_unittest(
    ucontext_test
    SUITE
      libc_ucontext_unittests
    SRCS
      ucontext_test.cpp
    DEPENDS
      libc.src.ucontext.getcontext
      libc.src.ucontext.setcontext
      libc.src.signal.sigemptyset
      libc.src.signal.sigaddset
      libc.src.signal.sigprocmask
  )
endif()
