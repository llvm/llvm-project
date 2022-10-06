#TODO(michaelrj): split out the implementations from memory_utils
add_header_library(
  memory_utils
  HDRS
    utils.h
    elements.h
    bcmp_implementations.h
    bzero_implementations.h
    memcmp_implementations.h
    memcpy_implementations.h
    memset_implementations.h
  DEPS
    libc.src.__support.CPP.bit
)

add_header_library(
  memcpy_implementation
  HDRS
    memcpy_implementations.h
  DEPS
    .memory_utils
)

add_header_library(
  memcmp_implementation
  HDRS
    memcmp_implementations.h
  DEPS
    .memory_utils
)

add_header_library(
  memset_implementation
  HDRS
    memset_implementations.h
  DEPS
    .memory_utils
)

add_header_library(
  bzero_implementation
  HDRS
    bzero_implementations.h
  DEPS
    .memset_implementation
)
