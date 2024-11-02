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
if(LLVM_LIBC_FULL_BUILD AND libc.include.stdbit IN_LIST TARGET_PUBLIC_HEADERS)
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

add_libc_test(
  stdckdint_test
  SUITE
    libc_include_tests
  SRCS
    stdckdint_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.stdckdint_macros
)
