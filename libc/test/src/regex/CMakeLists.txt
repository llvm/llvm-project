if(LLVM_LIBC_FULL_BUILD AND LLVM_LIBC_ENABLE_EXPERIMENTAL_ENTRYPOINTS)
  add_libc_testsuite(libc-regex-tests)

  add_libc_test(
    regerror_test
    SUITE
      libc-regex-tests
    SRCS
      regerror_test.cpp
    DEPENDS
      libc.hdr.regex_macros
      libc.src.regex.regerror
  )

  add_libc_test(
    regex_basic_test
    SUITE
      libc-regex-tests
    SRCS
      regex_basic_test.cpp
    DEPENDS
      libc.hdr.regex_macros
      libc.hdr.types.regex_t
      libc.src.__support.CPP.scope
      libc.src.regex.regcomp
      libc.src.regex.regexec
      libc.src.regex.regfree
  )
endif()
