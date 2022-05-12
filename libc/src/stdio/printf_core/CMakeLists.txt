
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

add_header_library(
  string_writer
  HDRS
    string_writer.h
  DEPENDS
    libc.src.string.memory_utils.memcpy_implementation
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
  DEPENDS
    .writer
    .core_structs
)
