if(LLVM_LIBC_FULL_BUILD AND LLVM_LIBC_ENABLE_EXPERIMENTAL_ENTRYPOINTS)
  add_custom_target(libc_regex_unittests)

  add_libc_unittest(
    regerror_test
    SUITE
      libc_regex_unittests
    SRCS
      regerror_test.cpp
    DEPENDS
      libc.hdr.regex_macros
      libc.src.regex.regerror
  )

  add_libc_unittest(
    regex_basic_test
    SUITE
      libc_regex_unittests
    SRCS
      regex_basic_test.cpp
    DEPENDS
      libc.hdr.regex_macros
      libc.hdr.types.regex_t
      libc.src.regex.regcomp
      libc.src.regex.regexec
      libc.src.regex.regfree
  )
endif()
