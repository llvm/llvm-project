#TODO(michaelrj): split out the implementations from memory_utils
add_header_library(
  memory_utils
  HDRS
    bcmp_implementations.h
    bzero_implementations.h
    memcmp_implementations.h
    memcpy_implementations.h
    memset_implementations.h
    op_aarch64.h
    op_higher_order.h
    op_builtin.h
    op_generic.h
    op_x86.h
    utils.h
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
