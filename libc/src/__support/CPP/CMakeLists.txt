add_header_library(
  array
  HDRS
    Array.h
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
)

add_header_library(
  string_view
  HDRS
    StringView.h
)

add_header_library(
  type_traits
  HDRS
    TypeTraits.h
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

add_header_library(
  uint
  HDRS
    UInt.h
  DEPENDS
    libc.src.__support.CPP.array
)
