add_libc_test(
  expr_test
  SUITE
    libc-support-tests
  SRCS
    expr_test.cpp
  DEPENDS
    libc.src.__support.regex.regex_expr_pool
    libc.hdr.regex_macros
  # ExprPool allocates a 128KB hash table, which exceeds the 64KB limit
  # of the simple bump allocator used in hermetic tests.
  UNIT_TEST_ONLY
)
