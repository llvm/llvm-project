if((${LIBC_TARGET_OS} STREQUAL "linux") AND (${LIBC_TARGET_MACHINE} MATCHES "i386|x86_64"))
  add_libc_unittest(
    x86_long_double_test
    SRCS
      x86_long_double_test.cpp
    DEPENDS
      libc.include.math
      libc.utils.FPUtil.fputil
  )
endif()
