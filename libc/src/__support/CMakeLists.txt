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
  high_precision_decimal
  HDRS
    high_precision_decimal.h

)

add_header_library(
  str_conv_utils
  HDRS
    str_conv_utils.h
  DEPENDS
    .ctype_utils
    .high_precision_decimal
    libc.include.errno
    libc.src.errno.__errno_location
    libc.utils.CPP.standalone_cpp
)

add_header_library(
  integer_operations
  HDRS
    integer_operations.h
)

add_subdirectory(FPUtil)
