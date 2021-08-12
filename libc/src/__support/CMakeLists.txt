add_header_library(
  common
  HDRS
    common.h
    endian.h
    sanitizer.h
)

add_header_library(
  ctype_utils
  HDRS
    ctype_utils.h
)

add_header_library(
  integer_operations
  HDRS
    integer_operations.h
)

add_subdirectory(FPUtil)
