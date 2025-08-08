add_header_library(
  helper_macros
  HDRS
    helper_macros.h
)

add_header_library(
  libc_constraint_handler
  HDRS
    libc_constraint_handler.h
  DEPENDS
    .abort_handler_s
    libc.hdr.types.errno_t
)
