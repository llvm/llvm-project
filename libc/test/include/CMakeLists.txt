add_custom_target(libc_include_tests)

add_libc_test(
  sys_queue_test
  SUITE
    libc_include_tests
  SRCS
    sys/queue_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.sys_queue_macros
    libc.src.__support.char_vector
    libc.src.__support.CPP.string
  COMPILE_OPTIONS
    # This is needed because the __containerof macro uses statement expression.
    -Wno-gnu-statement-expression-from-macro-expansion
)

# stdbit_test only tests our generated stdbit.h, which is not generated in
# overlay mode.
message(STATUS "TARGET_PUBLIC_HEADERS is ${TARGET_PUBLIC_HEADERS}")
if(LLVM_LIBC_FULL_BUILD AND libc.include.stdbit IN_LIST TARGET_PUBLIC_HEADERS
    AND libc.src.stdbit.stdc_leading_zeros_uc IN_LIST TARGET_LLVMLIBC_ENTRYPOINTS
    AND libc.src.stdbit.stdc_leading_zeros_us IN_LIST TARGET_LLVMLIBC_ENTRYPOINTS
    AND libc.src.stdbit.stdc_leading_zeros_ui IN_LIST TARGET_LLVMLIBC_ENTRYPOINTS
    AND libc.src.stdbit.stdc_leading_zeros_ul IN_LIST TARGET_LLVMLIBC_ENTRYPOINTS
    AND libc.src.stdbit.stdc_leading_zeros_ull IN_LIST TARGET_LLVMLIBC_ENTRYPOINTS
    AND libc.src.stdbit.stdc_leading_ones_uc IN_LIST TARGET_LLVMLIBC_ENTRYPOINTS
    AND libc.src.stdbit.stdc_leading_ones_us IN_LIST TARGET_LLVMLIBC_ENTRYPOINTS
    AND libc.src.stdbit.stdc_leading_ones_ui IN_LIST TARGET_LLVMLIBC_ENTRYPOINTS
    AND libc.src.stdbit.stdc_leading_ones_ul IN_LIST TARGET_LLVMLIBC_ENTRYPOINTS
    AND libc.src.stdbit.stdc_leading_ones_ull IN_LIST TARGET_LLVMLIBC_ENTRYPOINTS)
  add_libc_test(
    stdbit_test
    SUITE
      libc_include_tests
    SRCS
      stdbit_test.cpp
    DEPENDS
      libc.include.llvm-libc-macros.stdbit_macros
      libc.include.stdbit
      # Intentionally do not depend on libc.src.stdbit.*. The include test is
      # simply testing the macros provided by stdbit.h, not the implementation
      # of the underlying functions which the type generic macros may dispatch
      # to.
  )
endif()
