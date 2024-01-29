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

add_libc_test(
  stdbit_test
  SUITE
    libc_include_tests
  SRCS
    stdbit_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.stdbit_macros
    libc.src.__support.CPP.limits
    libc.src.stdbit.stdc_leading_zeros_uc
    libc.src.stdbit.stdc_leading_zeros_ui
    libc.src.stdbit.stdc_leading_zeros_ul
    libc.src.stdbit.stdc_leading_zeros_ull
    libc.src.stdbit.stdc_leading_zeros_us
  COMPILE_OPTIONS
    # stdbit is full of type generic macros implemented via C11 _Generic.
    -Wno-c11-extensions
)
