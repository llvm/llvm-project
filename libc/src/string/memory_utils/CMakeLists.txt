add_header_library(
  memory_utils
  HDRS
    utils.h
    elements.h
    bcmp_implementations.h
    memcmp_implementations.h
    memcpy_implementations.h
    memset_implementations.h
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
