add_libc_unittest(
  parser_test
  SUITE
    libc_stdio_unittests
  SRCS
    parser_test.cpp
  DEPENDS
    libc.src.stdio.printf_core.parser
    libc.src.__support.arg_list
)

target_link_libraries(libc.test.src.stdio.printf_core.parser_test PRIVATE LibcPrintfHelpers)
