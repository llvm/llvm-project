add_libc_unittest(
  converter_test
  SUITE
    libc_time_unittests
  SRCS
    converter_test.cpp
  DEPENDS
    libc.src.time.strftime_core.converter
    libc.src.stdio.printf_core.writer
    libc.src.time.strftime_core.core_structs
)

add_libc_unittest(
  parser_test
  SUITE
    libc_time_unittests
  SRCS
    parser_test.cpp
  DEPENDS
    libc.src.time.strftime_core.parser
    libc.src.time.strftime_core.core_structs

)
