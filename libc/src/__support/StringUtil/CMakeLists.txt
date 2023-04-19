add_header_library(
  message_mapper
  HDRS
    message_mapper.h
  DEPENDS
    libc.src.__support.CPP.string_view
    libc.src.__support.CPP.optional
)

# The table maps depend on message_mapper.
add_subdirectory(tables)

add_object_library(
  error_to_string
  HDRS
    error_to_string.h
  SRCS
    error_to_string.cpp
  DEPENDS
    .message_mapper
    libc.src.errno.errno
    libc.src.__support.CPP.span
    libc.src.__support.CPP.string_view
    libc.src.__support.CPP.stringstream
    libc.src.__support.integer_to_string
    libc.src.__support.StringUtil.tables.error_table
)

add_object_library(
  signal_to_string
  HDRS
    signal_to_string.h
  SRCS
    signal_to_string.cpp
  DEPENDS
    .message_mapper
    libc.include.signal
    libc.src.__support.CPP.span
    libc.src.__support.CPP.string_view
    libc.src.__support.CPP.stringstream
    libc.src.__support.integer_to_string
    libc.src.__support.StringUtil.tables.signal_table
)
