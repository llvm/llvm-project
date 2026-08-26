# Targets with implemented src/string/memory_utils/op_XXX.h specializations.
if(LIBC_TARGET_ARCHITECTURE_IS_AARCH64 OR
  LIBC_TARGET_ARCHITECTURE_IS_X86_64 OR
  LIBC_TARGET_ARCHITECTURE_IS_X86 OR
  LIBC_TARGET_ARCHITECTURE_IS_ANY_RISCV)
  add_libc_test(
    utils_test
    SUITE
      libc-string-tests
    SRCS
      op_tests.cpp
      utils_test.cpp
    COMPILE_OPTIONS
      ${LIBC_COMPILE_OPTIONS_NATIVE}
    DEPENDS
      libc.hdr.stdint_proxy
      libc.src.__support.CPP.array
      libc.src.__support.CPP.cstddef
      libc.src.__support.CPP.span
      libc.src.__support.macros.properties.os
      libc.src.__support.macros.properties.types
      libc.src.__support.macros.sanitizer
      libc.src.string.memory_utils.memory_utils
  )
endif()
