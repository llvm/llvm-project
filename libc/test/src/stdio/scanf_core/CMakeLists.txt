add_libc_unittest(
  parser_test
  SUITE
    libc_stdio_unittests
  SRCS
    parser_test.cpp
  LINK_LIBRARIES
    LibcScanfHelpers
  DEPENDS
    libc.src.stdio.scanf_core.parser
    libc.src.stdio.scanf_core.core_structs
    libc.src.__support.CPP.string_view
    libc.src.__support.arg_list
)

if(NOT (TARGET libc.src.__support.File.file))
  # Not all platforms have a file implementation. If file is unvailable,
  # then we must skip all the parts that need file.
  return()
endif()

add_libc_unittest(
  string_reader_test
  SUITE
    libc_stdio_unittests
  SRCS
    string_reader_test.cpp
  DEPENDS
    libc.src.stdio.scanf_core.reader
    libc.src.stdio.scanf_core.string_reader
    libc.src.__support.CPP.string_view
)

add_libc_unittest(
  converter_test
  SUITE
    libc_stdio_unittests
  SRCS
    converter_test.cpp
  DEPENDS
    libc.src.stdio.scanf_core.reader
    libc.src.stdio.scanf_core.string_reader
    libc.src.stdio.scanf_core.converter
    libc.src.__support.CPP.string_view
)
