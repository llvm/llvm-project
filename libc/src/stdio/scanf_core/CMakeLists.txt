add_header_library(
  core_structs
  HDRS
    core_structs.h
  DEPENDS
    libc.src.__support.CPP.string_view
    libc.src.__support.CPP.bitset
    libc.src.__support.FPUtil.fp_bits
)

add_header_library(
  parser
  HDRS
    parser.h
  DEPENDS
    .core_structs
    libc.src.__support.arg_list
    libc.src.__support.ctype_utils
    libc.src.__support.str_to_integer
    libc.src.__support.CPP.bit
    libc.src.__support.CPP.bitset
    libc.src.__support.CPP.string_view
)

if(NOT (TARGET libc.src.__support.File.file))
  # Not all platforms have a file implementation. If file is unvailable,
  # then we must skip all the parts that need file.
  return()
endif()

add_object_library(
  scanf_main
  SRCS
    scanf_main.cpp
  HDRS
    scanf_main.h
  DEPENDS
    .parser
    .reader
    .converter
    .core_structs
    libc.src.__support.arg_list
)

add_object_library(
  string_reader
  SRCS
    string_reader.cpp
  HDRS
    string_reader.h
)

add_object_library(
  file_reader
  SRCS
    file_reader.cpp
  HDRS
    file_reader.h
  DEPENDS
    libc.src.__support.File.file
)

add_object_library(
  reader
  SRCS
    reader.cpp
  HDRS
    reader.h
  DEPENDS
    .string_reader
    .file_reader
)

add_object_library(
  converter
  SRCS
    converter.cpp
    string_converter.cpp
    int_converter.cpp
    float_converter.cpp
    ptr_converter.cpp
  HDRS
    converter.h
    converter_utils.h
    int_converter.h
    string_converter.h
    float_converter.h
    current_pos_converter.h
    ptr_converter.h
  DEPENDS
    .reader
    .core_structs
    libc.src.__support.common
    libc.src.__support.ctype_utils
    libc.src.__support.CPP.bitset
    libc.src.__support.CPP.string_view
    libc.src.__support.CPP.limits
    libc.src.__support.char_vector
    libc.src.__support.str_to_float
)

add_object_library(
  vfscanf_internal
  SRCS
    vfscanf_internal.cpp
  HDRS
    vfscanf_internal.h
  DEPENDS
    .reader
    .file_reader
    .scanf_main
    libc.src.__support.arg_list
)
