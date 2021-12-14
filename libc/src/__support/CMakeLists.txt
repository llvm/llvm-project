add_subdirectory(CPP)

add_header_library(
  common
  HDRS
    architectures.h
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
  str_to_integer
  HDRS
    str_to_integer.h
  DEPENDS
    .ctype_utils
    libc.include.errno
    libc.src.errno.__errno_location
    libc.src.__support.CPP.standalone_cpp
)

add_header_library(
  str_to_float
  HDRS
    str_to_float.h
  DEPENDS
    .str_to_integer
    .ctype_utils
    .high_precision_decimal
    libc.include.errno
    libc.src.errno.__errno_location
    libc.src.__support.CPP.standalone_cpp
    libc.src.__support.FPUtil.fputil
)

add_header_library(
  integer_operations
  HDRS
    integer_operations.h
)

add_subdirectory(FPUtil)
add_subdirectory(OSUtil)
