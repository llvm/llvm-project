add_header_library(
  darwin_error_converter
  HDRS
    error_converter.h
  DEPENDS
    libc.src.stdio.printf_core.core_structs
    libc.hdr.errno_macros
)
