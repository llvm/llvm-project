add_header_library(
  error_table
  HDRS
    error_table.h
  DEPENDS
  libc.src.__support.StringUtil.message_mapper
  libc.src.errno.errno
)

add_header_library(
  signal_table
  HDRS
    signal_table.h
  DEPENDS
  libc.src.__support.StringUtil.message_mapper
  libc.include.signal
)
