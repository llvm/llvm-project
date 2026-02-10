add_header_library(
  aligned_access
  HDRS
    aligned_access.h
  DEPENDS
    libc.src.__support.common
    libc.src.__support.CPP.type_traits
)

add_header_library(
  byte_per_byte
  HDRS
    byte_per_byte.h
  DEPENDS
    libc.src.__support.common
    libc.src.__support.CPP.type_traits
    libc.src.string.memory_utils.utils
)

add_header_library(
  builtin
  HDRS
    builtin.h
  DEPENDS
    libc.src.__support.common
)

add_header_library(
  inline_strlen
  HDRS
    inline_strlen.h
  DEPENDS
    libc.src.__support.common
    libc.src.__support.CPP.type_traits
)
