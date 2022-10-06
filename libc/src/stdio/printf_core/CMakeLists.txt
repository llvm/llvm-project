
add_header_library(
  core_structs
  HDRS
    core_structs.h
  DEPENDS
    libc.src.__support.CPP.string_view
    libc.src.__support.FPUtil.fp_bits
)

add_object_library(
  parser
  SRCS
    parser.cpp
  HDRS
    parser.h
  DEPENDS
    .core_structs
    libc.src.__support.arg_list
    libc.src.__support.ctype_utils
    libc.src.__support.str_to_integer
    libc.src.__support.CPP.bit
    libc.src.__support.CPP.string_view
)

add_object_library(
  string_writer
  SRCS
    string_writer.cpp
  HDRS
    string_writer.h
  DEPENDS
    libc.src.__support.CPP.string_view
    libc.src.string.memory_utils.memcpy_implementation
    libc.src.string.memory_utils.memset_implementation
    .core_structs
)

add_object_library(
  writer
  SRCS
    writer.cpp
  HDRS
    writer.h
)

add_object_library(
  converter
  SRCS
    converter.cpp
  HDRS
    converter.h
    converter_atlas.h
    converter_utils.h
    string_converter.h
    char_converter.h
    int_converter.h
    ptr_converter.h
    write_int_converter.h
    float_inf_nan_converter.h
    float_hex_converter.h
  DEPENDS
    .writer
    .core_structs
    libc.src.__support.CPP.limits
    libc.src.__support.CPP.span
    libc.src.__support.CPP.string_view
    libc.src.__support.FPUtil.fp_bits
    libc.src.__support.FPUtil.fenv_impl
    libc.src.__support.integer_to_string
)


add_object_library(
  printf_main
  SRCS
    printf_main.cpp
  HDRS
    printf_main.h
  DEPENDS
    .parser
    .converter
    .writer
    .core_structs
    libc.src.__support.arg_list
)

if(NOT (TARGET libc.src.__support.File.file))
  # Not all platforms have a file implementation. If file is unvailable,
  # then we must skip all file based printf sections.
  return()
endif()

add_object_library(
  file_writer
  SRCS
    file_writer.cpp
  HDRS
    file_writer.h
  DEPENDS
    libc.src.__support.File.file
    libc.src.__support.CPP.string_view
    libc.src.string.memory_utils.memset_implementation
    .core_structs
)

add_object_library(
  vfprintf_internal
  SRCS
    vfprintf_internal.cpp
  HDRS
    vfprintf_internal.h
  DEPENDS
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
    libc.src.__support.arg_list
    libc.src.stdio.printf_core.printf_main
    libc.src.stdio.printf_core.file_writer
    libc.src.stdio.printf_core.writer
)
