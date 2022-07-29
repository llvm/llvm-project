add_header_library(
  array
  HDRS
    Array.h
)

add_header_library(
  uint
  HDRS
    UInt.h
  DEPENDS
    .array
)

add_header_library(
  uint128
  HDRS
    UInt128.h
  DEPENDS
    .uint
)

add_header_library(
  array_ref
  HDRS
    ArrayRef.h
)

add_header_library(
  bit
  HDRS
    Bit.h
)

add_header_library(
  bitset
  HDRS
    Bitset.h
)

add_header_library(
  functional
  HDRS
    Functional.h
)

add_header_library(
  limits
  HDRS
    Limits.h
  DEPENDS
    .uint
)

add_header_library(
  string_view
  HDRS
    StringView.h
)

add_header_library(
  stringstream
  HDRS
    stringstream.h
  DEPENDS
    .array_ref
    .string_view
    libc.src.__support.integer_to_string
)

add_header_library(
  type_traits
  HDRS
    type_traits.h
  DEPENDS
    .uint
)

add_header_library(
  utility
  HDRS
    Utility.h
)

add_header_library(
  vector
  HDRS
    vector.h
  DEPENDS
    libc.include.stdlib
)

add_header_library(
  atomic
  HDRS
    atomic.h
)

add_header_library(
  blockstore
  HDRS
    blockstore.h
)

add_header_library(
  error
  HDRS
    error.h
)
