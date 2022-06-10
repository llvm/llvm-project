
add_header_library(
  core_structs
  HDRS
    core_structs.h
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
    libc.src.string.memory_utils.memset_implementation
)

add_object_library(
  string_writer
  SRCS
    string_writer.cpp
  HDRS
    string_writer.h
  DEPENDS
    libc.src.string.memory_utils.memcpy_implementation
)

add_object_library(
  file_writer
  SRCS
    file_writer.cpp
  HDRS
    file_writer.h
  DEPENDS
    libc.src.__support.File.file
)

add_object_library(
  writer
  SRCS
    writer.cpp
  HDRS
    writer.h
  DEPENDS
    libc.src.string.memory_utils.memset_implementation
)

add_object_library(
  converter
  SRCS
    converter.cpp
  HDRS
    converter.h
    converter_atlas.h
    string_converter.h
    char_converter.h
    int_converter.h
  DEPENDS
    .writer
    .core_structs
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
