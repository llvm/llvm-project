add_custom_target(libc-nl-types-tests)

add_libc_test(
  nl_types_test
  SUITE
    libc-nl-types-tests
  SRCS
    nl_types_test.cpp
  DEPENDS
    libc.include.llvm-libc-macros.nl_types_macros
    libc.include.llvm-libc-types.nl_catd
    libc.src.nl_types.catopen
    libc.src.nl_types.catclose
    libc.src.nl_types.catgets
)
