add_header_library(
  baremetal_error_converter
  HDRS
    error_converter.h
  DEPENDS
    libc.src.stdio.printf_core.core_structs
    libc.hdr.errno_macros
)
