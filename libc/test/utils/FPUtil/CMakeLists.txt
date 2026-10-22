if((${LIBC_TARGET_OS} STREQUAL "linux") AND ((${LIBC_TARGET_ARCHITECTURE_IS_X86})
  OR (${LIBC_TARGET_ARCHITECTURE_IS_X86_64})))
  add_libc_test(
    x86_long_double_test
    SUITE
      libc-fputil-tests
    SRCS
      x86_long_double_test.cpp
    DEPENDS
      libc.hdr.math_macros
      libc.src.__support.FPUtil.fp_bits
  )
endif()
