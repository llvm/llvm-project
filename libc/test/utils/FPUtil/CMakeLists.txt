if((${LIBC_TARGET_OS} STREQUAL "linux") AND (${LIBC_TARGET_ARCHITECTURE_IS_X86}))
  add_libc_unittest(
    x86_long_double_test
    SRCS
      x86_long_double_test.cpp
    DEPENDS
      libc.hdr.math_macros
      libc.src.__support.FPUtil.fp_bits
  )
endif()
