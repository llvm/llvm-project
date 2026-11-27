add_libc_test(
  expr_test
  # Iterative alternation keeps spine buffers on the stack, so even small
  # coalescing cases can exceed the default 3 KiB NVPTX device stack.
  ENV
    LIBOMPTARGET_STACK_SIZE=8192
  SUITE
    libc-support-tests
  HDRS
    expr_builder.h
  SRCS
    expr_test.cpp
  DEPENDS
    libc.hdr.regex_macros
    libc.hdr.stdint_proxy
    libc.hdr.types.size_t
    libc.src.__support.error_or
    libc.src.__support.macros.config
    libc.src.__support.regex.regex_expr_pool
)

add_libc_test(
  matcher_test
  # The long grouped-repeat and RFC grammar cases need more than the default
  # 3 KiB device stack for recursive parsing.
  ENV
    LIBOMPTARGET_STACK_SIZE=65536
  SUITE
    libc-support-tests
  SRCS
    matcher_test.cpp
  DEPENDS
    libc.hdr.regex_macros
    libc.src.__support.CPP.string_view
    libc.src.__support.error_or
    libc.src.__support.regex.regex_expr_pool
    libc.src.__support.regex.regex_internal
    libc.src.__support.regex.regex_matcher
    libc.src.__support.regex.regex_parser
)

add_libc_test(
  matcher_limits_test
  SUITE
    libc-support-tests
  HDRS
    expr_builder.h
  SRCS
    matcher_limits_test.cpp
  DEPENDS
    libc.hdr.regex_macros
    libc.hdr.stdint_proxy
    libc.hdr.types.size_t
    libc.src.__support.error_or
    libc.src.__support.macros.config
    libc.src.__support.regex.regex_expr_pool
    libc.src.__support.regex.regex_matcher
    libc.src.__support.regex.regex_parser
)
