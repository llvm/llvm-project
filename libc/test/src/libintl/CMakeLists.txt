if(LLVM_LIBC_FULL_BUILD AND LLVM_LIBC_ENABLE_EXPERIMENTAL_ENTRYPOINTS)
  add_custom_target(libc-libintl-tests)

  add_libc_test(
    libintl_test
    SUITE
      libc-libintl-tests
    SRCS
      libintl_test.cpp
    DEPENDS
      libc.src.libintl.textdomain
      libc.test.UnitTest.ErrnoCheckingTest
      libc.test.UnitTest.ErrnoSetterMatcher
  )
endif()
