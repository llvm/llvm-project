if(LLVM_LIBC_FULL_BUILD)
  add_custom_target(libc_regex_unittests)

  add_libc_unittest(
    regerror_test
    SUITE
      libc_regex_unittests
    SRCS
      regerror_test.cpp
    DEPENDS
      libc.src.regex.regerror
  )

  add_libc_unittest(
    regex_basic_test
    SUITE
      libc_regex_unittests
    SRCS
      regex_basic_test.cpp
    DEPENDS
      libc.src.regex.regcomp
      libc.src.regex.regexec
      libc.src.regex.regfree
  )
endif()
