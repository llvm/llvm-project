add_header_library(
  core_structs
  HDRS
    core_structs.h
  DEPENDS
    libc.src.__support.CPP.string_view
    libc.include.time
)

add_header_library(
  parser
  HDRS
    parser.h
  DEPENDS
    .core_structs
    libc.src.string.string_utils
    libc.include.time

)

add_object_library(
  converter
  SRCS
    converter.cpp
  HDRS
    converter.h
    num_converter.h
    str_converter.h
    composite_converter.h
  DEPENDS
    .core_structs
    libc.src.__support.arg_list
    libc.src.stdio.printf_core.writer
    libc.src.stdio.printf_core.printf_main
    libc.src.__support.big_int
    libc.src.__support.CPP.string_view
    libc.src.__support.float_to_string
    libc.src.__support.integer_to_string
    libc.src.__support.uint128
    libc.src.__support.StringUtil.error_to_string
    libc.include.time

)

add_object_library(
  strftime_main
  SRCS
    strftime_main.cpp
  HDRS
    strftime_main.h
  DEPENDS
    .core_structs
    .parser
    .converter
    libc.src.stdio.printf_core.writer
    libc.include.time
)